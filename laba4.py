import numpy as np
import matplotlib.pyplot as plt

#функции p(t), q(t), f(t)
def p(t):
    return 0*t

def q(t):
    return 1 + 0*t

def f_rhs(t):
    # пример: f(t)=sin(pi*t)
    return np.sin(np.pi*t)

#Базисные функции φ_i и их конечно-разностные L(φ_i)
def phi(i, t):
    return np.cos(i * np.pi * (t + 1) / 2)

def L_phi(i, t_nodes, h_fd):
    """
    Возвращает вектор L(φ_i) на узлах t_nodes,
    где L(x)=x''+p(t)x'+q(t)x, 
    x',x'' аппроксимируются КР-формулами.
    """
    phi_i = phi(i, t_nodes)
    phi_i_m = phi(i, t_nodes - h_fd)
    phi_i_p = phi(i, t_nodes + h_fd)
    # вторая производная
    d2 = (phi_i_m - 2*phi_i + phi_i_p) / h_fd**2
    # первая производная
    d1 = (phi_i_p - phi_i_m) / (2*h_fd)
    return d2 + p(t_nodes)*d1 + q(t_nodes)*phi_i

#Правило Симпсона
def simpson_weights(a, b, M):
    """
    Возвращает массив весов для composite Simpson на [a,b]
    с M отрезками (M должно быть чётным).
    """
    if M % 2 == 1:
        M += 1
    h = (b - a) / M
    w = np.ones(M+1)
    w[1:-1:2] = 4
    w[2:-2:2] = 2
    return w * h / 3.0

#Cобираем СЛАУ A·C = b
def assemble_system(N, a, b, M_int):
    # Разбиение для интеграла
    t = np.linspace(a, b, M_int+1)
    w = simpson_weights(a, b, M_int)
    # шаг для конечных разностей
    h_fd = (b - a) / M_int
    
    A = np.zeros((N, N))
    b_vec = np.zeros(N)
    f_vals = f_rhs(t)
    
    for i in range(N):
        phi_i_vals = phi(i+1, t)
        # правая часть b_i = ∫ f(t) φ_i(t) dt
        b_vec[i] = np.dot(w, f_vals * phi_i_vals)
        for j in range(N):
            Lpj = L_phi(j+1, t, h_fd)
            A[i, j] = np.dot(w, Lpj * phi_i_vals)
    return A, b_vec

def main():
    print("Решение BVP методом Галеркина для φ_i(t)=cos(iπ(t+1)/2)")
    N = int(input("Введите число базисных функций N: "))
    
    # параметры области
    a, b = -1.0, 1.0
    # число шагов для интегрирования Simpson
    M_int = 200  # чётное, можно менять
    
    A, b_vec = assemble_system(N, a, b, M_int)
    C = np.linalg.solve(A, b_vec)
    
    # создаём плотную сетку для построения графика x_N(t)
    t_plot = np.linspace(a, b, 500)
    xN = np.zeros_like(t_plot)
    for j in range(N):
        xN += C[j] * phi(j+1, t_plot)
    
    # вывод результата
    print("Коэффициенты C_i:", np.round(C, 6))
    
    plt.figure(figsize=(8, 4))
    plt.plot(t_plot, xN, 'b-', label=r'$x_N(t)$')
    plt.title(f'Приближённое решение (N={N})')
    plt.xlabel('t')
    plt.ylabel('x_N(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()