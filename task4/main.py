import numpy as np
import sympy as sm
import matplotlib.pyplot as plt

# Параметры задачи

a = 0.0
b = 1.0
N = 10
h = (b - a) / N
lmb = -1.0

x_symb, y_symb = sm.symbols("x y")


# Ядро и правая часть

def K_num(x, y):
    """Числовое ядро K(x, y)."""
    return np.exp(x * y) / 5.0


def K_sym(x, y):
    """Символьное ядро K(x, y)."""
    return sm.exp(x * y) / 5.0


def f_num(x):
    """Правая часть в численном виде."""
    return 1.0 - x**2


# для совместимости с исходным кодом
f = f_num

# правая часть в символьном виде
f_symb = 1 - x_symb**2


# Решение через разложение ядра (выделение конечного ранга)

def compute_sol(K_expansion, rank):
    """
    Строит приближённое решение уравнения Фредгольма с ядром,
    заданным полиномом K_expansion(x, y), ранг = rank.

    Возвращает:
      u_expr   — символьное выражение решения u(x),
      u_values — значения u(x_i) в узлах x_i на сетке из N+1 точек.
    """
    # Разложение по y: K(x, y) = sum alpha_i(x) * ( ... y^k ... )
    poly_y = sm.Poly(K_expansion, y_symb)
    alpha_all = poly_y.all_coeffs()
    alpha = [c for c in alpha_all if c != 0]

    # Разложение по x: K(x, y) = sum beta_j(y) * ( ... x^k ... )
    poly_x = sm.Poly(K_expansion, x_symb)
    beta_all = poly_x.all_coeffs()
    beta = []
    for c in beta_all:
        if c == 0:
            continue
        norm = c.subs(y_symb, 1)  # нормировка
        beta.append(c / norm)

    alpha_y = [c.subs(x_symb, y_symb) for c in alpha]

    print("Output of alpha and beta")
    print("alpha:", alpha)
    print("beta :", beta)

    # Матрица гамма_{ij} = ∫_a^b beta_i(y) * alpha_j(y) dy
    gamma = np.zeros((rank, rank), dtype=float)
    for i in range(rank):
        for j in range(rank):
            integrand = beta[i] * alpha_y[j]
            gamma[i, j] = float(sm.integrate(integrand, (y_symb, a, b)))

    # Вектор B_i = ∫_a^b beta_i(y) * f(y) dy
    B = np.zeros(rank, dtype=float)
    for i in range(rank):
        integrand = beta[i] * f_symb.subs(x_symb, y_symb)
        B[i] = float(sm.integrate(integrand, (y_symb, a, b)))

    # (I - λγ) * C = B
    A = np.eye(rank) - lmb * gamma
    C = np.linalg.solve(A, B)

    # Восстановление решения u(x) = f(x) + λ * sum C_i * alpha_i(x)
    u_expr = f_symb
    for i in range(rank):
        u_expr += lmb * C[i] * alpha[i]

    # Значения на сетке
    u_values = np.zeros(N + 1, dtype=float)
    for i in range(N + 1):
        x_val = a + i * h
        u_values[i] = float(u_expr.subs(x_symb, x_val))

    print("Approximate solution u(x):", u_expr)
    return u_expr, u_values


# Квадратурная формула Гаусса

def compute_Gauss(N_nodes):
    """
    Решает задачу с помощью квадратурной формулы Гаусса
    с N_nodes узлами.

    Возвращает:
      u_values — значения u(x_i) в узлах x_i на сетке из N+1 точек.
    """
    # Полином Лежандра
    leg_poly_expr = sm.legendre(N_nodes, x_symb)
    leg_poly = sm.Poly(leg_poly_expr, x_symb)

    # Корни на [-1, 1]
    T_roots = np.array([float(root) for root in leg_poly.nroots()])
    T_roots = np.sort(T_roots)

    # Производная полинома Лежандра
    der_legendre_expr = sm.diff(leg_poly_expr, x_symb)
    der_legendre = sm.Poly(der_legendre_expr, x_symb)

    # Весовые коэффициенты A[i] на [-1, 1]
    A = np.zeros(N_nodes, dtype=float)
    for i in range(N_nodes):
        xi = T_roots[i]
        der_val = float(der_legendre.eval(xi))
        A[i] = 2.0 / ((1.0 - xi**2) * der_val**2)

    # Переход к [a, b]
    X = 0.5 * (a + b) + 0.5 * (b - a) * T_roots
    A = A * 0.5 * (b - a)

    # Система (I - λ K_quadrature) z = f(X)
    D = np.eye(N_nodes, dtype=float)
    for j in range(N_nodes):
        for k in range(N_nodes):
            D[j, k] -= lmb * A[k] * K_num(X[j], X[k])

    g = np.array([f_num(X[i]) for i in range(N_nodes)], dtype=float)
    z = np.linalg.solve(D, g)

    print(f"Vector z for N_nodes={N_nodes}:", z)

    # Восстановление u(x) ≈ f(x) + λ * sum A[i] K(x, X[i]) z[i]
    u_Gauss = f_symb
    for i in range(N_nodes):
        u_Gauss += lmb * A[i] * K_sym(x_symb, X[i]) * z[i]

    u_values = np.zeros(N + 1, dtype=float)
    for i in range(N + 1):
        x_val = a + i * h
        u_values[i] = float(u_Gauss.subs(x_symb, x_val))

    return u_values

if __name__ == "__main__":
    # Разложения ядра K(x, y) конечного ранга
    K_3 = (1 / 5) + (1 / 5) * x_symb * y_symb + (1 / 10) * x_symb**2 * y_symb**2
    u_3, u_3_values = compute_sol(K_3, rank=3)

    K_4 = K_3 + (1 / 30) * x_symb**3 * y_symb**3
    u_4, u_4_values = compute_sol(K_4, rank=4)

    print("u_3_values:", u_3_values)
    print("u_4_values:", u_4_values)

    # Сравнение решений ранга 3 и 4
    max_diff = np.max(np.abs(u_4_values - u_3_values))
    print("Max difference between u_3 and u_4:", max_diff)

    xh = np.linspace(a, b, N + 1)

    # График u_3 и u_4
    plt.figure(figsize=(12, 6))
    plt.plot(xh, u_3_values, label="u_3(x)", marker="o")
    plt.plot(xh, u_4_values, label="u_4(x)", marker="x")
    plt.title("Solutions u_3(x) and u_4(x)")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # График разности |u_4 - u_3|
    plt.figure(figsize=(12, 6))
    plt.plot(xh, np.abs(u_4_values - u_3_values), label="|u_4(x) - u_3(x)|", marker="o")
    plt.title("Difference between u_4(x) and u_3(x)")
    plt.xlabel("x")
    plt.ylabel("|u_4(x) - u_3(x)|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Решение с помощью формулы Гаусса
    u_Gauss_3_values = compute_Gauss(3)
    u_Gauss_4_values = compute_Gauss(4)

    print("u_Gauss_3_values:", u_Gauss_3_values)
    print("u_Gauss_4_values:", u_Gauss_4_values)

    # График решений по Гауссу
    plt.figure(figsize=(12, 6))
    plt.plot(xh, u_Gauss_3_values, label="u_Gauss_3(x) (3 nodes)", marker="o")
    plt.plot(xh, u_Gauss_4_values, label="u_Gauss_4(x) (4 nodes)", marker="x")
    plt.title("Solution using Gauss quadrature formula")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Разности между решениями по разложению и по Гауссу
    diff_3 = np.abs(u_3_values - u_Gauss_3_values)
    diff_4 = np.abs(u_4_values - u_Gauss_4_values)

    max_diff_3 = np.max(diff_3)
    max_diff_4 = np.max(diff_4)

    print(f"Max difference for rank-3 vs Gauss-3: {max_diff_3}")
    print(f"Max difference for rank-4 vs Gauss-4: {max_diff_4}")
    print(
        f"Max difference (Gauss-3 и Gauss-4): "
        f"{np.max(np.abs(u_Gauss_4_values - u_Gauss_3_values))}"
    )

    # График сравнения методов
    plt.figure(figsize=(12, 6))
    plt.plot(xh, diff_3, label="|u_3(x) - u_Gauss_3(x)|", marker="o")
    plt.plot(xh, diff_4, label="|u_4(x) - u_Gauss_4(x)|", marker="x")
    plt.title("Comparison of kernel expansion and Gauss quadrature solutions")
    plt.xlabel("x")
    plt.ylabel("Absolute difference")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()