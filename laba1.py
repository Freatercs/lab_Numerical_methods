import numpy as np
import matplotlib.pyplot as plt


def f(x, t):
    return np.sin(x) * np.cos(t)


def integrate(a, b, t, N, func):
    match func:
        case 1:
            return I1(a, b, t, N)
        case 2:
            return I2(a, b, t, N)
        case 3:
            return I3(a, b, t, N)
        case 4:
            return I4(a, b, t, N)
        case 5:
            return I5(a, b, t, N)


def I1(t, a, b, N):
    h = (b - a) / N
    integral_sum = 0.0

    for i in range(N):
        x_i = a + i * h
        x_ip1 = a + (i + 1) * h

        s_n = f(x_i, t) * (x_ip1 - x_i)
        integral_sum += s_n
    return integral_sum


def I2(t, a, b, N):
    h = (b - a) / N
    integral_sum = 0.0

    for i in range(N):
        x_i = a + i * h
        x_ip1 = a + (i + 1) * h

        s_n = f(x_ip1, t) * (x_ip1 - x_i)
        integral_sum += s_n
    return integral_sum


def I3(t, a, b, N):
    h = (b - a) / N
    integral_sum = 0.0

    for i in range(N):
        x_i = a + i * h
        x_ip1 = a + (i + 1) * h

        s_n = f((x_i + x_ip1) / 2, t) * (x_ip1 - x_i)
        integral_sum += s_n
    return integral_sum


def I4(t, a, b, N):
    h = (b - a) / N
    integral_sum = 0.0

    for i in range(N):
        x_i = a + i * h
        x_ip1 = a + (i + 1) * h

        s_n = (f(x_i, t) + f(x_ip1, t) / 2) * (x_ip1 - x_i)
        integral_sum += s_n
    return integral_sum


def I5(t, a, b, N):
    h = (b - a) / N
    integral_sum = 0.0

    for i in range(N):
        x_i = a + i * h
        x_ip1 = a + (i + 1) * h

        S_n = (
            (f(x_i, t) + 4 * f((x_i + x_ip1) / 2, t) + f(x_ip1, t)) / 6 * (x_ip1 - x_i)
        )
        integral_sum += S_n

    return integral_sum


def runge_rule(a, b, t, epsilon, way):
    N = 2  # Начальное количество подотрезков
    integral_old = integrate(a, b, t, N, way)

    while True:
        N *= 2 
        integral_new = integrate(a, b, t, N, way)

        # Оценка погрешности по правилу Рунге
        error_estimate = abs(integral_new - integral_old) / 3

        if error_estimate < epsilon:
            break

        integral_old = integral_new

    return integral_new


def main():
    a = float(input("Введите нижний предел a: "))
    b = float(input("Введите верхний предел b: "))
    alpha = float(input("Введите α: "))
    beta = float(input("Введите β: "))
    epsilon = float(input("Введите ε (точность): "))
    way = float(input("Введите вариант вычисления (1 - 5): "))
    if way < 1 or way > 5:
        way = 5

    t_values = np.linspace(alpha, beta, 100)
    I_values = [runge_rule(a, b, t, epsilon, way) for t in t_values]

    # Построение графика
    plt.plot(t_values, I_values, label="I(t) = ∫[a,b] f(x,t) dx", color="blue")
    plt.title("График функции I(t)")
    plt.xlabel("t")
    plt.ylabel("I(t)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
