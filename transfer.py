import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MU = 1


def rhs(t, STATE):
    # Распаковка из массива
    # STATE = [r, v, psi_r, psi_v]
    r = STATE[:3]
    v = STATE[3:6]
    psi_r = STATE[6:9]
    psi_v = STATE[9:12]

    drdt = v
    dvdt = - MU * r / (np.linalg.norm(r) ** 3) + psi_v
    dpsi_rdt = MU / (np.linalg.norm(r) ** 3) * psi_v - 3 * MU / (np.linalg.norm(r) ** 5) * np.outer(r, r) @ psi_v
    dpsi_vdt = - psi_r
    
    return np.hstack([drdt, dvdt, dpsi_rdt, dpsi_vdt], dtype=float)
    

def objective(w: np.ndarray) -> float:
    psi_r0, psi_v0 = w[:3], w[3:]

    r0 = np.array([1, 0, 0], dtype=float)
    v0 = np.array([0, 1, 0], dtype=float)

    X0 = np.hstack([r0, v0, psi_r0, psi_v0])

    THETA = 3 * np.pi / 4
    T = 1.35 * THETA

    rF = 1.5 * np.array([np.cos(THETA), np.sin(THETA), 0])
    vF = 1 / np.sqrt(1.5) * np.array([np.cos(THETA + np.pi / 2), np.sin(THETA + np.pi / 2), 0])

    # Интервал интегрирования
    t_span = (0, T)  # от 0 до T секунд
    t_eval = np.linspace(t_span[0], t_span[1], 200)  # точки для сохранения решения

    sol = solve_ivp(rhs, t_span, X0, t_eval=t_eval, method="DOP853", options={'rtol': 1e-14, 'atol': 1e-14})

    rlast = sol.y.T[-1][:3]
    vlast = sol.y.T[-1][3:6]

    res = np.dot(rlast - rF, rlast - rF) + np.dot(vlast - vF, vlast - vF)
    print(f"res = {res}")

    return res


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
    axs[0].set_title('Траектория "Орешника"')
    axs[0].set_xlabel('x [км]')
    axs[0].set_ylabel('y [км]')
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].minorticks_on()
    axs[0].grid(which='minor', color='gray', linestyle=':', linewidth=0.5)

    # Второй график: скорости vx и vy по времени
    axs[1].plot(t_eval, vx_array, label='vx')
    axs[1].plot(t_eval, vy_array, label='vy')
    axs[1].set_title('Скорости "Орешника"')
    axs[1].set_xlabel('Время [с]')
    axs[1].set_ylabel('Скорость [м / с]')
    axs[1].legend()
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].minorticks_on()
    axs[1].grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


plt.plot

res =  minimize(objective, np.array([0.0473, 0.0026, 0, 0.0038, 0.0556, 0]), method='Powell')

THETA = 3 * np.pi / 4
# Интервал интегрирования
t_span = (0, 1.35 * THETA)  # от 0 до T секунд
t_eval = np.linspace(t_span[0], t_span[1], 200)  # точки для сохранения решения

r0 = np.array([1, 0, 0], dtype=float)
v0 = np.array([0, 1, 0], dtype=float)\

X0 = np.hstack([r0, v0, res.x[:3], res.x[3:]])

sol = solve_ivp(rhs, t_span, X0, t_eval=t_eval, method="DOP853", options={'rtol': 1e-14, 'atol': 1e-14})

x_array = sol.y[0]
y_array = sol.y[1]

plt.plot(x_array, y_array)
plt.scatter(1.5 * np.array([np.cos(THETA), np.sin(THETA)])[0], 1.5 * np.array([np.cos(THETA), np.sin(THETA)])[1])
plt.axis('equal')
plt.grid(True)
plt.show()

print(f'f = {res.x}')