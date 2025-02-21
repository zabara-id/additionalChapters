import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Параметры задачи
A = 60.     # Ускорение от ракеты (пример)
G = 9.81     # Ускорение свободного падения
H = 400_000  # Высота, м

X0 = np.array([0, 0, 0, 0], dtype=float)  # НУ [x, y, vx, vy]
T = 300  # сек


def rhs(t, Y, u_0, c):
    # Распаковка из массива
    x, y, vx, vy = Y
    
    # Вычисляем u(t)
    u = np.arctan(np.tan(u_0) + c*t)

    cosu = 1 / np.sqrt(np.tan(u)**2 + 1)
    sinu = np.tan(u) / np.sqrt(np.tan(u)**2 + 1)
    
    # Формируем правые части
    dxdt = vx
    dydt = vy
    dvxdt = A * cosu
    dvydt = A * sinu - G
    
    return np.array([dxdt, dydt, dvxdt, dvydt], dtype=float)


def objective(var: np.ndarray) -> float:
    u_0, c = var

    # Интервал интегрирования
    t_span = (0, T)  # от 0 до T секунд
    t_eval = np.linspace(t_span[0], t_span[1], 200)  # точки для сохранения решения

    sol = solve_ivp(rhs, t_span, X0, args=(u_0, c), t_eval=t_eval)

    y_array = sol.y[1]
    vy_array = sol.y[3]

    y_last = y_array[-1]
    vy_last = vy_array[-1]

    return (y_last - H) ** 2 + vy_last**2


def get_trajectory(var: np.ndarray):
    u_0, c = var

    # Интервал интегрирования
    t_span = (0, T)  # от 0 до 100 секунд
    t_eval = np.linspace(t_span[0], t_span[1], 200)

    # Запуск решения уравнения методом Адамса
    sol = solve_ivp(rhs, t_span, X0, args=(u_0, c), t_eval=t_eval)


    x_array = sol.y[0]
    y_array = sol.y[1]
    vx_array = sol.y[2]
    vy_array = sol.y[3]

    # Создание фигуры с двумя subplot'ами
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    axs[0].plot(x_array, y_array)
    axs[0].set_title('Траектория движения')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].minorticks_on()
    axs[0].grid(which='minor', color='gray', linestyle=':', linewidth=0.5)

    # Второй график: скорости vx и vy по времени
    axs[1].plot(t_eval, vx_array, label='vx')
    axs[1].plot(t_eval, vy_array, label='vy')
    axs[1].set_title('Скорости по времени')
    axs[1].set_xlabel('Время')
    axs[1].set_ylabel('Скорость')
    axs[1].legend()
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].minorticks_on()
    axs[1].grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


res =  minimize(objective, [np.pi/4, 0], method='nelder-mead')

print(f"(u*, c*) = {res.x}")

get_trajectory(res.x)