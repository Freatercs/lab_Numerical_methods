import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify


def f(x, y):
    # Пример: двойной квадратичный колодец
    return (x - 1)**2 + 2*(y - 2)**2

# Градиент
x_sym, y_sym = symbols('x y')
f_expr = f(x_sym, y_sym)
df_dx_expr = diff(f_expr, x_sym)
df_dy_expr = diff(f_expr, y_sym)

df_dx = lambdify((x_sym, y_sym), df_dx_expr, 'numpy')
df_dy = lambdify((x_sym, y_sym), df_dy_expr, 'numpy')

def golden_section_search(phi, a, b, eps):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > eps:
        if phi(c) <= phi(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (a + b) / 2

def coordinate_descent(x0, y0, eps, a, b, c, d):
    path = [(x0, y0)]
    xk, yk = x0, y0
    while True:
        phi_x = lambda x: f(x, yk)
        xk1 = golden_section_search(phi_x, a, b, eps)
        
        phi_y = lambda y: f(xk1, y)
        yk1 = golden_section_search(phi_y, c, d, eps)
        path.append((xk1, yk1))
        if np.hypot(xk1 - xk, yk1 - yk) < eps:
            break
        xk, yk = xk1, yk1
    return path


def steepest_descent(x0, y0, eps):
    path = [(x0, y0)]
    xk, yk = x0, y0
    while True:
        gx = df_dx(xk, yk)
        gy = df_dy(xk, yk)
        phi_alpha = lambda alpha: f(xk - alpha * gx, yk - alpha * gy)
        alpha_k = golden_section_search(phi_alpha, 0, 1, eps)
        xk1 = xk - alpha_k * gx
        yk1 = yk - alpha_k * gy
        path.append((xk1, yk1))
        if np.hypot(xk1 - xk, yk1 - yk) < eps:
            break
        xk, yk = xk1, yk1
    return path

def plot_contour_and_path(path, a, b, c, d, levels=30):
    X = np.linspace(a, b, 200)
    Y = np.linspace(c, d, 200)
    XX, YY = np.meshgrid(X, Y)
    ZZ = f(XX, YY)

    plt.figure(figsize=(8, 6))
    plt.contour(XX, YY, ZZ, levels=levels, cmap='viridis')
    px, py = zip(*path)
    plt.plot(px, py, 'ro-', label='Поиск минимума')
    plt.scatter(px[0], py[0], c='green', label='Начало')
    plt.scatter(px[-1], py[-1], c='red', label='Конец')
    plt.xlim(a, b)
    plt.ylim(c, d)
    plt.legend()
    plt.title('Контурные линии и траектория оптимизации')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def main():
    print("Введите через пробел a b c d (область определения f):")
    a, b, c, d = map(float, input().split())
    eps = float(input("Точность ε = "))
    x0, y0 = map(float, input("Начальная точка x0 y0 = ").split())
    print("Выберите метод: 1 — покоординатный, 2 — наискорейший градиентный")
    method = input().strip()

    if method == '1':
        path = coordinate_descent(x0, y0, eps, a, b, c, d)
    else:
        path = steepest_descent(x0, y0, eps)

    x_min, y_min = path[-1]
    print(f"Найденная точка минимума: x = {x_min:.6f}, y = {y_min:.6f}")
    print(f"Значение функции в этой точке: f = {f(x_min, y_min):.6f}")

    plot_contour_and_path(path, a, b, c, d)

if __name__ == "__main__":
    main()