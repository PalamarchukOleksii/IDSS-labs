import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

A = 0.4
B = 0.8
C = 0.5
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

def compute_gradient():
    a, b, c, sigma, w1, w2 = sp.symbols("a b c sigma w1 w2")
    r_xd = sp.Matrix([a, b])
    R_x = sp.Matrix([[1, c], [c, 1]])
    w = sp.Matrix([w1, w2])
    
    E = 1/2 * sigma**2 - (r_xd.T * w)[0] + 1/2 * (w.T * R_x * w)[0]
    grad_E = sp.Matrix([sp.diff(E, w1), sp.diff(E, w2)])
    
    return sp.lambdify((w1, w2, a, b, c), grad_E, 'numpy')

def loss_function(w1, w2, a, b, c):
    return 0.5 * (w1**2 + w2**2 + 2 * c * w1 * w2) - a * w1 - b * w2

def gradient_descent(a_val, b_val, c_val, eta, tol=None, T=1000):
    grad_func = compute_gradient()
    w_num = np.array([0.0, 0.0])
    trajectory = [w_num.copy()]
    loss_values = []
    
    for t in range(T):
        grad_val = np.array(grad_func(w_num[0], w_num[1], a_val, b_val, c_val)).flatten()
        step = eta * grad_val
        w_num = w_num - step
        trajectory.append(w_num.copy())
        loss_values.append(loss_function(w_num[0], w_num[1], a_val, b_val, c_val))
        
        if t % 10 == 0:
            print(f"Iteration {t}: w = {w_num}")
        
        if tol is not None and np.linalg.norm(step) < tol:
            break
        
    print(f"Final w* on iter {t}: {w_num}")
    return np.array(trajectory), np.array(loss_values)

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

def plot_3d_loss(trajectory, a, b, c, eta, filename):
    w1_vals = np.linspace(-2, 2, 5)
    w2_vals = np.linspace(-2, 2, 5)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    E_vals = loss_function(W1, W2, a, b, c)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W1, W2, E_vals, cmap='Blues', alpha=0.7)
    ax.plot(trajectory[:, 0], trajectory[:, 1], loss_function(trajectory[:, 0], trajectory[:, 1], a, b, c), 'r-', label='path')
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel("E(w)")
    ax.set_title(f"3D Loss Function and Path (η={eta})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

def plot_loss_vs_epoch(loss_values, eta, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_values, label="E(w)")
    plt.xlabel("Epoch")
    plt.ylabel("E(w)")
    plt.title(f"Loss Function Over Iterations (η={eta})")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

def plot_loss_vs_w(trajectory, loss_values, eta, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(trajectory[1:, 0], loss_values, label="E(w)")  # Фікс: Виключаємо першу точку
    plt.xlabel("w")
    plt.ylabel("E(w)")
    plt.title(f"Loss Function vs w (η={eta})")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

# Виконуємо gradient descent для двох випадків
eta_converge = 0.1
eta_diverge = 2.0  # Попередньо підібране значення для розходження

# Для сходження
print("Converging case:")
trajectory1, loss_values1 = gradient_descent(A, B, C, eta_converge, tol=0.0001)
plot_3d_loss(trajectory1, A, B, C, eta_converge, "3d_converging.png")
plot_loss_vs_epoch(loss_values1, eta_converge, "loss_vs_epoch_converging.png")
plot_loss_vs_w(trajectory1, loss_values1, eta_converge, "loss_vs_w_converging.png")
save_trajectory_plot(trajectory1, eta_converge, f"Gradient Descent with η = {eta_converge} (Converging)", "converging_eta_0.1.png")

# Для розходження
print("Diverging case:")
trajectory2, loss_values2 = gradient_descent(A, B, C, eta_diverge, tol=0.0001)
plot_3d_loss(trajectory2, A, B, C, eta_diverge, "3d_diverging.png")
plot_loss_vs_epoch(loss_values2, eta_diverge, "loss_vs_epoch_diverging.png")
plot_loss_vs_w(trajectory2, loss_values2, eta_diverge, "loss_vs_w_diverging.png")
save_trajectory_plot(trajectory2, eta_diverge, f"Gradient Descent with η = {eta_diverge} (Diverging)", "diverging_eta_2.0.png")
