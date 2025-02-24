import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os

# Створюємо папку для збереження зображень, якщо вона не існує
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

A = 0.4
B = 0.8
C = 0.5

def compute_gradient():
    a, b, c, sigma, w1, w2 = sp.symbols("a b c sigma w1 w2")
    r_xd = sp.Matrix([a, b])
    R_x = sp.Matrix([[1, c], [c, 1]])
    w = sp.Matrix([w1, w2])
    
    E = 1/2 * sigma**2 - (r_xd.T * w)[0] + 1/2 * (w.T * R_x * w)[0]
    grad_E = sp.Matrix([sp.diff(E, w1), sp.diff(E, w2)])
    
    return sp.lambdify((w1, w2, a, b, c), grad_E, 'numpy')

def gradient_descent(a_val, b_val, c_val, eta, tol=None, T=1000):
    grad_func = compute_gradient()
    w_num = np.array([0.0, 0.0])
    trajectory = [w_num.copy()]
    
    for t in range(T):
        grad_val = np.array(grad_func(w_num[0], w_num[1], a_val, b_val, c_val)).flatten()
        step = eta * grad_val
        w_num = w_num - step
        trajectory.append(w_num.copy())
        
        if t % 10 == 0:
            print(f"Iteration {t}: w = {w_num}")
        
        if tol is not None and np.linalg.norm(step) < tol:
            break
    
    print(f"Final w* on iter {t}: {w_num}")
    return np.array(trajectory)

def save_trajectory_plot(trajectory, eta, title, filename):
    trajectory = np.array(trajectory)
    fig = plt.figure(figsize=(12, 5))
    
    # 3D plot w1, w2, t
    ax1 = fig.add_subplot(121, projection='3d')
    t_vals = np.arange(len(trajectory))
    ax1.plot3D(trajectory[:, 0], trajectory[:, 1], t_vals, marker='o', linestyle='-')
    ax1.set_xlabel("w1")
    ax1.set_ylabel("w2")
    ax1.set_zlabel("Iteration t")
    ax1.set_title(f"Trajectory of w(t) in 3D (η = {eta})")
    
    # 2D plot w1 vs w2
    ax2 = fig.add_subplot(122)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-')
    ax2.set_xlabel("w1")
    ax2.set_ylabel("w2")
    ax2.set_title(f"Trajectory in W-plane (η = {eta})")
    ax2.grid()
    
    plt.suptitle(title)
    
    # Збереження зображення в файл
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Plot saved as {filepath}")

# Виконуємо gradient descent для декількох значень ETA
eta_converge = 0.1  # Приклад значення, яке сходиться
eta_diverge = 2.0  # Приклад значення, яке може розходитися

# Для сходження
print("Converging case:")
trajectory1 = gradient_descent(A, B, C, eta_converge, tol=0.0001)
save_trajectory_plot(trajectory1, eta_converge, f"Gradient Descent with η = {eta_converge} (Converging)", "converging_eta_0.1.png")

# Для розходження
print("Diverging case:")
trajectory2 = gradient_descent(A, B, C, eta_diverge, tol=0.0001)
save_trajectory_plot(trajectory2, eta_diverge, f"Gradient Descent with η = {eta_diverge} (Diverging)", "diverging_eta_2.0.png")
