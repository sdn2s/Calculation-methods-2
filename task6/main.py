import time
import math

import numpy as np
import scipy.sparse as sp

a = 0.0
b = 1.0


def p(x: float) -> float:
    return 1.0


def r(x: float) -> float:
    return 2.0 * x


def f_rhs(x: float) -> float:
    # правая часть исходной краевой задачи
    return (3.0 * x * x - 1.0) / ((x * x + 1.0) ** 3)


def q(x: float) -> float:
    return x * x + 1.0


def integ_phi_i(f_fun, x: float, h: float) -> float:
    """Интеграл f(x)*phi_i(x) ~ f в средней точке * площадь треугольника."""
    return f_fun(x + h) * h


def integ_phi_ii(f_fun, x: float, h: float) -> float:
    """Интеграл f(x)*phi_i(x)^2."""
    return (2.0 / 3.0) * f_fun(x + h) * h


def integ_phi_ij(f_fun, x: float, h: float) -> float:
    """Интеграл f(x)*phi_i(x)*phi_{i+1}(x)."""
    return (1.0 / 6.0) * f_fun(x + 1.5 * h) * h


def integ_der_ii(f_fun, x: float, h: float) -> float:
    """Интеграл f(x)*phi_i'(x)^2."""
    return (f_fun(x + 0.5 * h) + f_fun(x + 1.5 * h)) / h


def integ_der_ij(f_fun, x: float, h: float) -> float:
    """Интеграл f(x)*phi_i'(x)*phi_{i+1}'(x)."""
    return -f_fun(x + 1.5 * h) / h


def build_stiffness_matrix(p_fun, r_fun, N: int):
    """
    Строим матрицу жесткости K (N-1 x N-1) по КЭ-задаче из задачи 1.
    Для задачи на собственные значения правая часть не нужна.
    """
    h = 1.0 / N
    B = np.zeros(N - 1)
    A = np.zeros(N - 2)

    for i in range(N - 1):
        B[i] = (
            integ_phi_ii(r_fun, i * h, h)
            + integ_der_ii(p_fun, i * h, h)
        )

    for i in range(N - 2):
        A[i] = (
            integ_phi_ij(r_fun, i * h, h)
            + integ_der_ij(p_fun, i * h, h)
        )

    K = sp.lil_matrix((N - 1, N - 1))
    for i in range(N - 1):
        K[i, i] = B[i]
    for i in range(N - 2):
        K[i, i + 1] = A[i]
        K[i + 1, i] = A[i]

    return K.tocsr()


def power_method(
    K: sp.csr_matrix,
    eps: float = 1e-7,
    max_iter: int = 1_000_000,
    y0=None,
):
    """
    Степенной метод для поиска наибольшего по модулю собственного значения матрицы K.
    K — csr_matrix (n x n).

    Возвращает:
      lambda_max, eigenvector, iterations, residual_norm, elapsed_time.
    """
    n = K.shape[0]

    if y0 is None:
        # Улучшенный выбор начального вектора: y0_j = (-1)^j, j = 1,...,n
        # Это увеличивает проекцию на собственный вектор при lambda_max
        v = np.fromiter(((-1.0) ** (j + 1) for j in range(n)), dtype=float)
    else:
        v = np.asarray(y0, dtype=float).copy()
        if v.shape[0] != n:
            raise ValueError("Начальный вектор y0 неверной длины")

    v /= np.linalg.norm(v)
    lambda_old = 0.0
    start = time.time()

    for it in range(1, max_iter + 1):
        w = K.dot(v)
        # скалярное произведение <v, Kv> / <v, v>
        lambda_new = float(np.dot(w, v) / np.dot(v, v))
        # невязка
        res_vec = w - lambda_new * v
        res_norm = np.linalg.norm(res_vec) / np.linalg.norm(v)
        if res_norm < eps:
            elapsed = time.time() - start
            return lambda_new, v / np.linalg.norm(v), it, res_norm, elapsed
        v = w / np.linalg.norm(w)
        lambda_old = lambda_new

    elapsed = time.time() - start
    return lambda_old, v / np.linalg.norm(v), max_iter, res_norm, elapsed


def power_method_min_eigen(
    K: sp.csr_matrix,
    eps: float = 1e-7,
    max_iter: int = 1_000_000,
    y0=None,
):
    """
    Степенной метод для минимального собственного числа:
    применяем степенной метод к матрице K^{-1}, но явно обратную не строим,
    а на каждом шаге решаем систему K w = v.

    Возвращает:
      lambda_min, eigenvector, iterations, residual_norm, elapsed_time.
    """
    n = K.shape[0]
    K_dense = K.toarray()  # один раз

    if y0 is None:
        v = np.ones(n, dtype=float)
    else:
        v = np.asarray(y0, dtype=float).copy()
        if v.shape[0] != n:
            raise ValueError("Начальный вектор y0 неверной длины")

    v /= np.linalg.norm(v)
    mu_old = 0.0  # собственное число для K^{-1}
    start = time.time()

    for it in range(1, max_iter + 1):
        # решаем K w = v  => w = K^{-1} v
        w = np.linalg.solve(K_dense, v)
        # собственное значение для K^{-1}
        mu_new = float(np.dot(w, v) / np.dot(v, v))
        v_new = w / np.linalg.norm(w)
        if np.linalg.norm(v_new - v) / np.linalg.norm(v) < eps:
            lambda_min = 1.0 / mu_new
            res_vec = K_dense.dot(v_new) - lambda_min * v_new
            res_norm = np.linalg.norm(res_vec) / np.linalg.norm(v_new)
            elapsed = time.time() - start
            return lambda_min, v_new, it, res_norm, elapsed
        v = v_new
        mu_old = mu_new

    lambda_min = 1.0 / mu_old
    res_vec = K_dense.dot(v) - lambda_min * v
    res_norm = np.linalg.norm(res_vec) / np.linalg.norm(v)
    elapsed = time.time() - start
    return lambda_min, v, max_iter, res_norm, elapsed


def jacobi_eigen(K: np.ndarray, eps: float = 1e-5, max_iter: int = 100_000):
    """
    Классический метод Якоби для симметричной матрицы K (плотной).
    Возвращает:
      eigenvalues, eigenvectors, iterations.
    """
    A = np.array(K, dtype=float)
    n = A.shape[0]
    V = np.eye(n, dtype=float)

    def max_offdiag(a):
        """Ищем максимальный по модулю наддиагональный элемент."""
        max_val = 0.0
        p = 0
        q = 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i, j]) > max_val:
                    max_val = abs(a[i, j])
                    p, q = i, j
        return max_val, p, q

    it = 0
    while it < max_iter:
        max_val, p, q = max_offdiag(A)
        if max_val < eps:
            break
        it += 1

        # угол поворота
        if A[p, p] == A[q, q]:
            phi = math.pi / 4.0
        else:
            phi = 0.5 * math.atan2(2.0 * A[p, q], A[q, q] - A[p, p])

        c = math.cos(phi)
        s = math.sin(phi)

        # строим матрицу поворота R (ортогональную)
        R = np.eye(n)
        R[p, p] = c
        R[q, q] = c
        R[p, q] = -s
        R[q, p] = s

        # A_{k+1} = R^T A_k R
        A = R.T @ A @ R
        # V = V R (накопление собственных векторов)
        V = V @ R

    eigenvalues = np.diag(A)
    eigenvectors = V
    return eigenvalues, eigenvectors, it


def residuals_for_eigenpairs(
    K: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
):
    """
    Для каждой пары (λ_i, e_i) считает норму невязки ||K e_i - λ_i e_i||_2.
    Возвращает массив невязок той же длины.
    """
    n = K.shape[0]
    res = np.zeros(n, dtype=float)
    for i in range(n):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        r_vec = K @ v - lam * v
        res[i] = np.linalg.norm(r_vec)
    return res


if __name__ == "__main__":
    # 1. Построение матрицы жесткости
    # Для полной задачи (Якоби) берём умеренный N, чтобы метод был не слишком тяжёлым.
    N = 30
    K_sparse = build_stiffness_matrix(p, r, N)
    K_dense = K_sparse.toarray()
    n_dim = K_dense.shape[0]

    print(f"Размер матрицы K: {n_dim} x {n_dim}")

    # 2. Частичная задача: степенной метод (максимальное и минимальное собственные числа)
    print("\nСтепенной метод (максимальное собственное число)")
    lam_max, v_max, it_max, res_max, time_max = power_method(K_sparse, eps=1e-7)
    print(f"lambda_max ≈ {lam_max:.7f}")
    print(f"Число итераций: {it_max}")
    print(f"Норма невязки: {res_max:.3e}")
    print(f"Время работы: {time_max:.3f} c")

    print("\nСтепенной метод (минимальное собственное число через K^{-1})")
    lam_min, v_min, it_min, res_min, time_min = power_method_min_eigen(K_sparse, eps=1e-7)
    print(f"lambda_min ≈ {lam_min:.7f}")
    print(f"Число итераций: {it_min}")
    print(f"Норма невязки: {res_min:.3e}")
    print(f"Время работы: {time_min:.3f} c")

    # 3. Полная задача: метод Якоби
    print("\nМетод Якоби для полной задачи собственных значений")
    eigenvalues, eigenvectors, it_jac = jacobi_eigen(K_dense, eps=1e-5)
    res = residuals_for_eigenpairs(K_dense, eigenvalues, eigenvectors)

    # сортируем по возрастанию собственных чисел
    idx = np.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[idx]
    res_sorted = res[idx]

    print(f"Число итераций Якоби: {it_jac}")
    print("\nТаблица собственных чисел и норм невязки (первые 10):")
    print("№".center(4), "λ".center(18), "||K e - λ e||".center(18))
    for k in range(min(10, n_dim)):
        print(
            f"{k+1:>2d} {eigenvalues_sorted[k]:18.8f} {res_sorted[k]:18.6e}"
        )
