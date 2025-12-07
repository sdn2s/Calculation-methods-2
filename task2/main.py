import numpy as np
import pandas as pd


def exact_solution(x, t):
    return np.sin(2 * t + 1) * np.cos(2 * x)


def f(x, t):
    return (2 * np.cos(2 * t + 1) + 4 * np.sin(2 * t + 1) + np.sin(x) * np.sin(2 * t + 1)) * np.cos(2 * x)


def g(x):
    return np.sin(1) * np.cos(2 * x)


def alpha(t):
    return np.sin(2 * t + 1)


def beta(t):
    return np.sin(2 * t + 1) * (np.cos(2) + 2 * np.sin(2))


def a(x, t):
    return 1.0


def b(x, t):
    return 0.0


def c(x, t):
    return 0.0


def progonka(K, F):
    """
    Метод прогонки (Томаса) для трёхдиагональной системы:
        K * y = F
    K — квадратная матрица (numpy.ndarray) размера n x n,
    F — вектор длины n.
    Возвращает вектор y длины n.
    """
    K = np.asarray(K, dtype=float)
    F = np.asarray(F, dtype=float)
    n = K.shape[0]

    s = np.zeros(n - 1)
    t = np.zeros(n)
    y = np.zeros(n)

    # прямой ход
    s[0] = -K[0, 1] / K[0, 0]
    t[0] = F[0] / K[0, 0]

    for i in range(1, n - 1):
        denom = K[i, i] + K[i, i - 1] * s[i - 1]
        s[i] = -K[i, i + 1] / denom
        t[i] = (F[i] - K[i, i - 1] * t[i - 1]) / denom

    denom_last = K[n - 1, n - 1] + K[n - 1, n - 2] * s[n - 2]
    t[n - 1] = (F[n - 1] - K[n - 1, n - 2] * t[n - 2]) / denom_last

    # обратный ход
    y[n - 1] = t[n - 1]
    for i in range(n - 2, -1, -1):
        y[i] = s[i] * y[i + 1] + t[i]

    return y


def yav(f_fun, g_fun, a_fun, N, beta_fun, alpha_fun):
    """
    Явная схема:
      - шаг по x: h = 1/N
      - шаг по t: tau = h^2/8
      - T = 0.1
    """
    h = 1.0 / N
    tau = h**2 / 8.0
    T = 0.1
    M2 = int(T / tau)

    u = np.zeros((M2 + 1, N + 1))

    # начальное условие
    for i in range(N + 1):
        u[0, i] = g_fun(h * i)

    # шаги по времени
    for j in range(1, M2 + 1):
        t_prev = tau * (j - 1)
        t_cur = tau * j

        # левое граничное условие
        u[j, 0] = alpha_fun(t_cur)

        # внутренние узлы
        for i in range(1, N):
            x_i = h * i

            Lu = a_fun(x_i, t_prev) * (u[j - 1, i + 1] - 2 * u[j - 1, i] + u[j - 1, i - 1]) / (h**2)
            u[j, i] = u[j - 1, i] + tau * (Lu + f_fun(x_i, t_prev))

        # правое граничное условие (по условию задачи)
        u[j, N] = beta_fun(t_cur)

    return u


def neyav(f_fun, g_fun, a_fun, N, M, beta_fun, alpha_fun, sigma):
    """
    Неявная разностная схема с весами:
      - шаг по x: h = 1/N
      - шаг по t: tau = 0.1/M
    sigma — параметр схемы (0, 0.5, 1 и т.д.).
    """
    h = 1.0 / N
    tau = 0.1 / M

    u2 = np.zeros((M + 1, N + 1))

    # начальное условие
    for i in range(N + 1):
        u2[0, i] = g_fun(h * i)

    # шаги по времени
    for j in range(1, M + 1):
        t_prev = tau * (j - 1)
        t_cur = tau * j

        A = np.zeros((N + 1, N + 1))
        d = np.zeros(N + 1)

        for i in range(N + 1):
            if i == 0:
                # левое граничное условие
                d[i] = -alpha_fun(t_cur)
                A[i, i] = -1.0
                A[i, i + 1] = 0.0
            elif i == N:
                # правое граничное условие
                d[i] = -beta_fun(t_cur)
                A[i, i - 1] = 0.0
                A[i, i] = -1.0
            else:
                x_i = h * i

                Lu1 = a_fun(x_i, t_prev) * (u2[j - 1, i + 1] - 2 * u2[j - 1, i] + u2[j - 1, i - 1]) / (h**2)
                d[i] = -u2[j - 1, i] / tau - (1.0 - sigma) * Lu1 - f_fun(x_i, t_prev + sigma * tau)

                A[i, i] = -((1.0 / tau) + sigma * (2 * a_fun(x_i, t_cur)) / (h**2) - sigma * c(x_i, t_cur))

                A[i, i - 1] = sigma * (a_fun(x_i, t_cur) / (h**2) - b(x_i, t_cur) / (2 * h))
                A[i, i + 1] = sigma * (a_fun(x_i, t_cur) / (h**2) + b(x_i, t_cur) / (2 * h))

        # решаем трёхдиагональную систему
        u2[j, :] = progonka(A, d)
        # при желании можно использовать np.linalg.solve(A, d)

    return u2


def matr_to_6x6(u):
    """
    Берёт матрицу решения u(t_i, x_j) и выбирает из неё 6x6
    значений, равномерно распределённых по времени и по x.
    """
    u_out = np.zeros((6, 6))
    M1 = u.shape[0] - 1
    N1 = u.shape[1] - 1

    step_t = int(M1 / 5)
    step_x = int(N1 / 5)

    for i in range(6):
        for j in range(6):
            u_out[i, j] = u[i * step_t, j * step_x]

    return u_out


if __name__ == "__main__":
    # Ввод N_ex: количество промежутков по x (обычно кратно 5: 5, 10, 20, ...)
    N_ex = int(input("Введите количество промежутков по x (например, 5, 10, 20): "))

    h_ex = 1.0 / N_ex
    tau_ex = h_ex**2 / 8.0
    T = 0.1
    M_ex = int(T / tau_ex)

    # точное решение на сетке (по введённому N_ex)
    u_exact = np.zeros((M_ex + 1, N_ex + 1))
    for i in range(N_ex + 1):
        u_exact[0, i] = exact_solution(h_ex * i, 0.0)
    for j in range(1, M_ex + 1):
        t_j = tau_ex * j
        for i in range(N_ex + 1):
            x_i = h_ex * i
            u_exact[j, i] = exact_solution(x_i, t_j)

    # явная схема
    u = yav(f, g, a, N_ex, beta, alpha)

    # неявная схема при sigma = 0
    sigma = 0.0
    u2 = neyav(f, g, a, N_ex, M_ex, beta, alpha, sigma)

    # 6x6 выборки
    u_out = matr_to_6x6(u)
    u2_out = matr_to_6x6(u2)
    u_ex_out = matr_to_6x6(u_exact)

    # таблицы
    index_t = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
    columns_x = [0, 0.2, 0.4, 0.6, 0.8, 1]

    table_u = pd.DataFrame(u_out, index=index_t, columns=columns_x)
    table_u2 = pd.DataFrame(u2_out, index=index_t, columns=columns_x)
    table_u_ex = pd.DataFrame(u_ex_out, index=index_t, columns=columns_x)

    # сходимость явной схемы
    shod_yav = np.zeros((3, 4))
    shod_yav[0, 0] = 0.2
    shod_yav[1, 0] = 0.1
    shod_yav[2, 0] = 0.05

    shod_yav[0, 1] = 0.2**2 / 8.0
    shod_yav[1, 1] = 0.1**2 / 8.0
    shod_yav[2, 1] = 0.05**2 / 8.0

    # ВНИМАНИЕ: здесь, как и в оригинальном коде, для u_exact берётся сетка,
    # построенная по введённому N_ex, а не по 5, 10, 20.
    shod_yav[0, 2] = np.max(np.abs(matr_to_6x6(u_exact) - matr_to_6x6(yav(f, g, a, 5, beta, alpha))))
    shod_yav[1, 2] = np.max(np.abs(matr_to_6x6(u_exact) - matr_to_6x6(yav(f, g, a, 10, beta, alpha))))
    shod_yav[2, 2] = np.max(np.abs(matr_to_6x6(u_exact) - matr_to_6x6(yav(f, g, a, 20, beta, alpha))))

    shod_yav[1, 3] = np.max(np.abs(matr_to_6x6(yav(f, g, a, 10, beta, alpha)) - matr_to_6x6(yav(f, g, a, 5, beta, alpha))))
    shod_yav[2, 3] = np.max(np.abs(matr_to_6x6(yav(f, g, a, 20, beta, alpha)) - matr_to_6x6(yav(f, g, a, 10, beta, alpha))))

    table_shod = pd.DataFrame(
        shod_yav,
        index=[1, 2, 3],
        columns=["h", "tau", "||u_ex - u(h,tau)||", "||u(2h,tau) - u(h,tau)||"],
    )

    print("Сходимость явной схемы")
    print(table_shod)

    # сходимость неявной схемы при разных sigma
    for KJ in range(0, 3):
        sigma_neyav = 0.5 * KJ
        shod_neyav = np.zeros((3, 4))

        shod_neyav[0, 0] = 0.2
        shod_neyav[1, 0] = 0.1
        shod_neyav[2, 0] = 0.05

        # шаги по времени для M = 20, 80, 320
        shod_neyav[0, 1] = 0.005       # 0.1 / 20
        shod_neyav[1, 1] = 0.1**2 / 8  # M = 80
        shod_neyav[2, 1] = 0.05**2 / 8  # M = 320

        shod_neyav[0, 2] = np.max(
            np.abs(
                matr_to_6x6(u_exact)
                - matr_to_6x6(neyav(f, g, a, 5, 20, beta, alpha, sigma_neyav))
            )
        )
        shod_neyav[1, 2] = np.max(
            np.abs(
                matr_to_6x6(u_exact)
                - matr_to_6x6(neyav(f, g, a, 10, 80, beta, alpha, sigma_neyav))
            )
        )
        shod_neyav[2, 2] = np.max(
            np.abs(
                matr_to_6x6(u_exact)
                - matr_to_6x6(neyav(f, g, a, 20, 320, beta, alpha, sigma_neyav))
            )
        )

        shod_neyav[1, 3] = np.max(
            np.abs(
                matr_to_6x6(neyav(f, g, a, 10, 80, beta, alpha, sigma_neyav))
                - matr_to_6x6(neyav(f, g, a, 5, 20, beta, alpha, sigma_neyav))
            )
        )
        shod_neyav[2, 3] = np.max(
            np.abs(
                matr_to_6x6(neyav(f, g, a, 20, 320, beta, alpha, sigma_neyav))
                - matr_to_6x6(neyav(f, g, a, 10, 80, beta, alpha, sigma_neyav))
            )
        )

        table_shod_ne = pd.DataFrame(
            shod_neyav,
            index=[1, 2, 3],
            columns=["h", "tau", "||u_ex - u(h,tau)||", "||u(2h,tau) - u(h,tau)||"],
        )

        print("Сходимость схемы c весами при sigma = " + str(sigma_neyav))
        print(table_shod_ne)

    print("Результаты вычислений для явной схемы N = " + str(N_ex))
    print(table_u)

    print("Результаты вычислений для неявной схемы N = " + str(N_ex) + " sigma = " + str(0.0))
    print(table_u2)

    # отдельно считаем неявную схему для sigma = 0.5 и 1
    for sigma_val in [0.5, 1.0]:
        u2_sigma = neyav(f, g, a, N_ex, M_ex, beta, alpha, sigma_val)
        u2_sigma_out = matr_to_6x6(u2_sigma)
        table_u2_sigma = pd.DataFrame(u2_sigma_out, index=index_t, columns=columns_x)
        print("Результаты вычислений для неявной схемы N = " + str(N_ex) + " sigma = " + str(sigma_val))
        print(table_u2_sigma)

    print("Таблица точного решения N = " + str(N_ex))
    print(table_u_ex)
