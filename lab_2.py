import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, symbols, lambdify

def f(x, mu):
    return x**2 + x**3 - x - mu

#Производная
x_sym, mu_sym = symbols('x mu')
f_expr = f(x_sym, mu_sym)
df_expr = diff(f_expr, x_sym)
df = lambdify((x_sym, mu_sym), df_expr, 'numpy')



def bisection_method(f, a, b, mu, epsilon):
    if f(a, mu) * f(b, mu) > 0:
        return None
    
    while (b - a) > epsilon:
        c = (a + b) / 2
        if abs(f(c, mu)) < epsilon:
            return c
        if f(a, mu) * f(c, mu) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def newton_method(f, df, x0, mu, epsilon):
    x = x0
    while True:
        x_new = x - f(x, mu) / df(x, mu)
        if abs(x_new - x) < epsilon:
            return x_new
        x = x_new

def find_all_roots(f, df, a, b, mu, epsilon, method='bisection'):
    roots = []
    
    # Разбиваем отрезок на подотрезки
    n = 100
    dx = (b - a) / n
    
    for i in range(n):
        x1 = a + i * dx
        x2 = x1 + dx
        
        if f(x1, mu) * f(x2, mu) <= 0:
            if method == 'bisection':
                root = bisection_method(f, x1, x2, mu, epsilon)
            else:  # newton
                root = newton_method(f, df, (x1 + x2) / 2, mu, epsilon)
            
            if root is not None:
                roots.append(root)
    
    return roots

def main():
    print("Введите границы отрезка [a,b]:")
    a = float(input("a = "))
    b = float(input("b = "))
    
    print("Введите границы параметра mu [alpha, beta]:")
    alpha = float(input("alpa = "))
    beta = float(input("beta = "))
    
    print("Выберите метод:")
    print("1. Метод деления отрезка пополам")
    print("2. Метод Ньютона")
    method_choice = input("Ваш выбор (1 или 2): ")
    
    epsilon = 1e-6
    
    mu_values = np.linspace(alpha, beta, 100)
    solutions = []
    
    for mu in mu_values:
        roots = find_all_roots(f, df, a, b, mu, epsilon, 
                             'bisection' if method_choice == '1' else 'newton')
        for root in roots:
            solutions.append((mu, root))
    
    # Построение графика
    if solutions:
        solutions = np.array(solutions)
        plt.figure(figsize=(10, 6))
        plt.plot(solutions[:, 0], solutions[:, 1], 'b.', label='Решения')
        plt.xlabel('μ')
        plt.ylabel('x')
        plt.title('Решения уравнения f(x,μ) = 0')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("Решения не найдены")

if __name__ == "__main__":
    main()