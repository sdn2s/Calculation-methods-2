import time
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


a = 0
b = 1


def p(x):
    return 1


def r(x):
    return 2 * x


def f(x):
    return (3 * x * x - 1) / ((x * x + 1) ** 3)


def q(x):
    return x**2 + 1


# ------------------------
# Интегралы от базисных функций
# ------------------------


def integ_phi_i(f_fun, x, h):
    """Интеграл f(x)*phi_i(x) ~ f в средней точке * площадь треугольника."""
    return f_fun(x + h) * h


def integ_phi_ii(f_fun, x, h):
    """Интеграл f(x)*phi_i(x)^2."""
    return (2 / 3) * f_fun(x + h) * h


def integ_phi_ij(f_fun, x, h):
    """Интеграл f(x)*phi_i(x)*phi_{i+1}(x)."""
    return (1 / 6) * f_fun(x + 1.5 * h) * h


def integ_der_ii(f_fun, x, h):
    """Интеграл f(x)*phi_i'(x)^2."""
    return (f_fun(x + 0.5 * h) + f_fun(x + 1.5 * h)) / h


def integ_der_ij(f_fun, x, h):
    """Интеграл f(x)*phi_i'(x)*phi_{i+1}'(x)."""
    return -f_fun(x + 1.5 * h) / h


# ------------------------
# Построение матрицы и правой части
# ------------------------


def coef(p_fun, r_fun, f_fun, N):
    """
    Строим матрицу жесткости K и вектор правой части F.
    Возвращаем:
      K  — csr_matrix, размер (N-1, N-1),
      F  — numpy.ndarray, длина N-1.
    """
    h = 1.0 / N
    B = np.zeros(N - 1)
    A = np.zeros(N - 2)

    # правая часть
    F = np.zeros(N - 1)

    for i in range(N - 1):
        F[i] = integ_phi_i(f_fun, i * h, h)

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

    return K.tocsr(), F


# ------------------------
# Оценка шага sigma для Якоби
# ------------------------


def define_sigma(K, N):
    """
    Подбор шага sigma по сумме элементов в столбцах
    """
    Kd = K.toarray()
    sum_of_elem = np.zeros(N - 1)

    sum_of_elem[0] = Kd[0, 0] + Kd[1, 0]
    for j in range(1, N - 2):
        sum_of_elem[j] = Kd[j, j] + Kd[j - 1, j] + Kd[j + 1, j]
    sum_of_elem[N - 2] = Kd[N - 2, N - 2] + Kd[N - 3, N - 2]

    return 2.0 / np.max(sum_of_elem)


# ------------------------
# Метод прогонки (трёхдиагональная система)
# ------------------------


def progonka(K, F, N):
    """
    Решение трёхдиагональной системы K y = F методом прогонки.
    K — csr_matrix (N-1, N-1),
    F — ndarray длины N-1.
    Возвращает:
      y — ndarray длины N-1,
      work_time — время работы.
    """
    start_time = time.time()

    Kd = K.toarray()
    Fd = np.asarray(F).ravel()
    n = N - 1

    s = np.zeros(n - 1)
    t = np.zeros(n)
    y = np.zeros(n)

    s[0] = -Kd[0, 1] / Kd[0, 0]
    t[0] = Fd[0] / Kd[0, 0]

    for i in range(1, n - 1):
        denom = Kd[i, i] + Kd[i, i - 1] * s[i - 1]
        s[i] = -Kd[i, i + 1] / denom

    for i in range(1, n):
        denom = Kd[i, i] + Kd[i, i - 1] * s[i - 1]
        t[i] = (Fd[i] - Kd[i, i - 1] * t[i - 1]) / denom

    y[n - 1] = t[n - 1]
    for i in range(n - 2, -1, -1):
        y[i] = s[i] * y[i + 1] + t[i]

    work_time = time.time() - start_time
    return y, work_time


# ------------------------
# Метод Якоби
# ------------------------


def Jacoby(K, G, err, maxNum, N, sigma):
    """
    Метод Якоби с шагом sigma.
    K — csr_matrix (N-1, N-1),
    G — ndarray длины N-1,
    err — критерий остановки по норме,
    maxNum — максимум итераций,
    N — параметр сетки,
    sigma — шаг.
    Возвращает:
      u_new — ndarray длины N-1,
      work_time — время работы,
      counter — число итераций.
    """
    start_time = time.time()

    n = N - 1
    D_inv = 1.0 / K.diagonal()
    G_arr = np.asarray(G).ravel()

    u_old = np.zeros(n)
    u_new = np.zeros(n)

    counter = 0
    for _ in range(maxNum):
        counter += 1
        u_old = u_new.copy()
        r = K.dot(u_old) - G_arr
        u_new = u_old - sigma * D_inv * r
        if np.linalg.norm(u_new - u_old) < err:
            break

    work_time = time.time() - start_time
    return u_new, work_time, counter


# ------------------------
# Матрица преобразования для декомпозиции
# ------------------------


def Transformation_matrix(N):
    """
    Матрица преобразования C (N-1 x N-1) с возвратом csr.
    """
    n = int(math.sqrt(N))
    C = sp.lil_matrix((N - 1, N - 1))

    # единичная
    for i in range(N - 1):
        C[i, i] = 1.0

    kn = 1
    for i in range(0, (n - 1) * (n - 1)):
        C[i, n * (n - 1) + i // (n - 1)] = kn / n
        kn += 1
        if kn == n:
            kn = 1

    kn = n - 1
    for i in range(n - 1, (n - 1) * n):
        C[i, n * (n - 1) + i // (n - 1) - 1] = kn / n
        kn -= 1
        if kn == 0:
            kn = n - 1

    return C.tocsr()


# ------------------------
# Метод декомпозиции
# ------------------------


def decomp(K, F, err, maxNum, N, sigma):
    """
    Метод декомпозиции (с предобуславливателем Delta_Hh).
    K — csr_matrix (N-1, N-1),
    F — ndarray длины N-1,
    err — критерий остановки,
    maxNum — максимум итераций,
    N — параметр сетки (должен быть квадратом n^2),
    sigma — шаг.
    Возвращает:
      u_new_global — ndarray длины N-1 (в исходном базисе),
      work_time — время работы,
      counter — число итераций.
    """
    start_time = time.time()

    n = int(math.sqrt(N))
    h = 1.0 / N

    # Предобуславливатель Delta_Hh
    Delta_Hh = sp.lil_matrix((N - 1, N - 1))
    Delta = sp.lil_matrix((N - 1, N - 1))

    # блоки для Delta
    for i in range(0, n * (n - 1)):
        Delta[i, i] = 2.0
    for i in range(0, n * (n - 1) - 2):
        Delta[i, i + 1] = -1.0
        Delta[i + 1, i] = -1.0
    Delta = Delta * (1.0 / h)

    # блоки для Delta_Hh (верхний диапазон индексов)
    for i in range(n * (n - 1), n * n - 1):
        Delta_Hh[i, i] = 2.0
    for i in range(n * (n - 1), n * n - 2):
        Delta_Hh[i, i + 1] = -1.0
        Delta_Hh[i + 1, i] = -1.0
    Delta_Hh = Delta_Hh / (h * n)

    Delta_Hh = Delta_Hh + Delta

    # среднее значение p(x) на [0,1] (как в исходном коде)
    p_arr = np.zeros(101)
    h1 = 1.0 / 100
    for i in range(101):
        p_arr[i] = p(i * h1)
    p_mean = np.mean(p_arr)

    Delta_Hh = (Delta_Hh * p_mean).tocsr()

    # Матрица преобразования
    C = Transformation_matrix(N)
    CT = C.transpose().tocsr()

    KDD = CT.dot(K.dot(C))
    FDD = CT.dot(F)

    u_old = np.zeros(N - 1)
    u_new = np.zeros(N - 1)
    counter = 0

    for _ in range(maxNum):
        u_old = u_new.copy()
        counter += 1

        d_k = KDD.dot(u_old) - FDD

        w_k, _ = progonka(Delta_Hh, sigma * d_k, N)
        u_new = u_old - w_k

        if np.linalg.norm(u_new - u_old) < err:
            break

    # обратно в исходный базис
    u_new_global = C.dot(u_new)
    u_new_global = np.asarray(u_new_global).ravel()

    work_time = time.time() - start_time
    return u_new_global, work_time, counter


# ------------------------
# Основной запуск
# ------------------------


if __name__ == "__main__":
    # N — количество элементов для прогонки и Якоби
    N = 100
    h = 1.0 / N

    K, F = coef(p, r, f, N)
    sigma = define_sigma(K, N)

    # Прогонка
    y_prog, time_prog = progonka(K, F, N)

    # Якоби
    y_jac, time_jac, jac_iter = Jacoby(K, F, 1e-4, 10000, N, sigma)

    xh = np.arange(h, 1.0, h)

    # N1 — количество элементов для метода декомпозиции
    N1 = 100
    h1 = 1.0 / N1

    K1, F1 = coef(p, r, f, N1)
    sigma1 = define_sigma(K1, N1)

    y_decomp, time_decomp, decomp_iter = decomp(K1, F1, 1e-4, 10000, N1, sigma1)
    xhD = np.arange(h1, 1.0, h1)

    # Таблица значений
    print("N = ", N, ":")
    print("Времена выполнения:")
    print(
        "Прогонка: ",
        time_prog,
        " Якоби: ",
        time_jac,
        " Декомпозиция: ",
        time_decomp,
        " сек",
    )
    print("Число итераций: Якоби =", jac_iter, ", декомпозиция =", decomp_iter, "\n")

    print("x".center(3), "U1(x)".center(15), "U2(x)".center(15), "U3(x)".center(15))
    for i in range(1, 10):
        idx = (N * i) // 10
        idx1 = (N1 * i) // 10
        print(
            "%2.1f %15.12f %15.12f %15.12f "
            % (
                i * 0.1,
                y_prog[idx],
                y_jac[idx],
                y_decomp[idx1],
            )
        )

    print("\n")
    print("№".center(7), "A".center(15), "B".center(17), "C".center(17), "F".center(17))

    # K в плотном виде для печати коэффициентов
    Kd_full = K.toarray()
    for i in range(1, 10):
        idx = (N * i) // 10
        A_val = Kd_full[idx, idx]
        B_val = Kd_full[idx, idx - 1] if idx - 1 >= 0 else 0.0
        C_val = Kd_full[idx, idx + 1] if idx + 1 < N - 1 else 0.0
        print(
            "%4d %16.12f %16.12f %16.12f %16.12f"
            % (
                idx,
                A_val,
                B_val,
                C_val,
                F[idx],
            )
        )

    # ------------------------
    # Графики решений для фиксированного N
    # ------------------------
    plt.figure(figsize=(12, 7))

    plt.subplot(1, 3, 1)
    plt.grid(True)
    plt.title("Прогонка N=" + str(N), fontsize=9)
    plt.xlabel("x", fontsize=15, color="gray")
    plt.ylabel("y(x)", fontsize=15, color="gray")
    plt.plot(xh, y_prog)

    plt.subplot(1, 3, 2)
    plt.grid(True)
    plt.title("Якоби N=" + str(N), fontsize=9)
    plt.xlabel("x", fontsize=15, color="gray")
    plt.plot(xh, y_jac)

    plt.subplot(1, 3, 3)
    plt.grid(True)
    plt.title("Декомпозиция N=" + str(N1), fontsize=9)
    plt.xlabel("x", fontsize=15, color="gray")
    plt.plot(xhD, y_decomp)

    plt.tight_layout()

    # ------------------------
    # Графики: погрешность от N и число итераций от N
    # ------------------------

    Ns = [25, 49, 100, 196, 400]  # квадраты, чтобы работала декомпозиция области

    errors_jac = []
    errors_decomp = []
    iters_jac = []
    iters_decomp = []

    for N_test in Ns:
        K_test, F_test = coef(p, r, f, N_test)
        sigma_test = define_sigma(K_test, N_test)

        # "Точное" решение для данного N — прогонка
        y_prog_test, _ = progonka(K_test, F_test, N_test)

        # Якоби
        y_jac_test, _, jac_iter_test = Jacoby(
            K_test, F_test, 1e-4, 10000, N_test, sigma_test
        )

        # Декомпозиция (N_test квадрат, как нужно)
        y_decomp_test, _, decomp_iter_test = decomp(
            K_test, F_test, 1e-4, 10000, N_test, sigma_test
        )

        # Погрешности относительно решения прогонкой
        err_j = np.linalg.norm(y_jac_test - y_prog_test)
        err_d = np.linalg.norm(y_decomp_test - y_prog_test)

        errors_jac.append(err_j)
        errors_decomp.append(err_d)
        iters_jac.append(jac_iter_test)
        iters_decomp.append(decomp_iter_test)

    # График погрешности от N
    plt.figure(figsize=(7, 5))
    plt.grid(True)
    plt.title("Зависимость погрешности от N")
    plt.xlabel("N")
    plt.ylabel("||u_iter - u_prog||_2")
    plt.plot(Ns, errors_jac, marker="o", label="Якоби")
    plt.plot(Ns, errors_decomp, marker="s", label="Декомпозиция")
    plt.legend()

    # График числа итераций от N
    plt.figure(figsize=(7, 5))
    plt.grid(True)
    plt.title("Зависимость числа итераций от N")
    plt.xlabel("N")
    plt.ylabel("Число итераций")
    plt.plot(Ns, iters_jac, marker="o", label="Якоби")
    plt.plot(Ns, iters_decomp, marker="s", label="Декомпозиция")
    plt.legend()

    plt.show()
