import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
A = 0.4
B = 0.8
C = 0.5

ETA_CONVERGE = 0.1
ETA_DIVERGE = 1.5

SAVED_PLOTS_PATH = "plots"
SAVE_PLOTS = True
SHOW_PLOTS = False

if SAVE_PLOTS:
    os.makedirs(SAVED_PLOTS_PATH, exist_ok=True)


def compute_gradient():
    a, b, c, sigma, w1, w2 = sp.symbols("a b c sigma w1 w2")
    r_xd = sp.Matrix([a, b])
    R_x = sp.Matrix([[1, c], [c, 1]])
    w = sp.Matrix([w1, w2])

    E = 1 / 2 * sigma**2 - (r_xd.T * w)[0] + 1 / 2 * (w.T * R_x * w)[0]
    grad_E = sp.Matrix([sp.diff(E, w1), sp.diff(E, w2)])

    return sp.lambdify((w1, w2, a, b, c), grad_E, "numpy")


def loss_function(w1, w2, a, b, c):
    return 0.5 * (w1**2 + w2**2 + 2 * c * w1 * w2) - a * w1 - b * w2


def gradient_descent(a_val, b_val, c_val, eta, tol=None, T=1000):
    grad_func = compute_gradient()
    w_num = np.array([0.0, 0.0])
    trajectory = [w_num.copy()]
    loss_values = []

    for t in range(T):
        grad_val = np.array(
            grad_func(w_num[0], w_num[1], a_val, b_val, c_val)
        ).flatten()
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


def save_plot_function(fig, filename):
    filepath = os.path.join(SAVED_PLOTS_PATH, filename)
    fig.savefig(filepath)
    print(f"Plot saved as {filepath}")
    plt.close(fig)


def save_trajectory_plot(trajectory, eta, title, filename):
    trajectory = np.array(trajectory)
    fig = plt.figure(figsize=(12, 5))

    # 3D plot w1, w2, t
    ax1 = fig.add_subplot(121, projection="3d")
    t_vals = np.arange(len(trajectory))
    ax1.plot3D(trajectory[:, 0], trajectory[:, 1], t_vals, marker="o", linestyle="-")
    ax1.set_xlabel("w1")
    ax1.set_ylabel("w2")
    ax1.set_zlabel("Iteration t")
    ax1.set_title(f"Trajectory of w(t) in 3D (η = {eta})")

    # 2D plot w1 vs w2
    ax2 = fig.add_subplot(122)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], marker="o", linestyle="-")
    ax2.set_xlabel("w1")
    ax2.set_ylabel("w2")
    ax2.set_title(f"Trajectory in W-plane (η = {eta})")
    ax2.grid()

    plt.suptitle(title)

    if SHOW_PLOTS:
        plt.show()

    if SAVE_PLOTS:
        save_plot_function(fig, filename)


def plot_3d_loss(trajectory, a, b, c, eta, filename):
    w1_vals = np.linspace(-2, 2, 5)
    w2_vals = np.linspace(-2, 2, 5)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    E_vals = loss_function(W1, W2, a, b, c)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(W1, W2, E_vals, cmap="Blues", alpha=0.7)
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        loss_function(trajectory[:, 0], trajectory[:, 1], a, b, c),
        "r-",
        label="path",
    )
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel("E(w)")
    ax.set_title(f"3D Loss Function and Path (η={eta})")
    ax.legend()

    if SHOW_PLOTS:
        plt.show()

    if SAVE_PLOTS:
        save_plot_function(fig, filename)


def plot_loss_vs_epoch(loss_values, eta, filename):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(loss_values, label="E(w)")
    plt.xlabel("Epoch")
    plt.ylabel("E(w)")
    plt.title(f"Loss Function Over Iterations (η={eta})")
    plt.legend()
    plt.grid()

    if SHOW_PLOTS:
        plt.show()

    if SAVE_PLOTS:
        save_plot_function(fig, filename)


def plot_loss_vs_w(trajectory, loss_values, eta, filename):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(trajectory[1:, 0], loss_values, label="E(w)")
    plt.xlabel("w")
    plt.ylabel("E(w)")
    plt.title(f"Loss Function vs w (η={eta})")
    plt.legend()
    plt.grid()

    if SHOW_PLOTS:
        plt.show()

    if SAVE_PLOTS:
        save_plot_function(fig, filename)


def run_experiment(a, b, c, eta, tol=0.0001, T=1000, experiment_name="experiment"):
    print(f"\n\nRunning experiment: {experiment_name} with η = {eta}")
    trajectory, loss_values = gradient_descent(a, b, c, eta, tol, T)

    plot_3d_loss(trajectory, a, b, c, eta, f"{experiment_name}_3d.png")
    plot_loss_vs_epoch(loss_values, eta, f"{experiment_name}_loss_vs_epoch.png")
    plot_loss_vs_w(trajectory, loss_values, eta, f"{experiment_name}_loss_vs_w.png")
    save_trajectory_plot(
        trajectory,
        eta,
        f"Gradient Descent (η = {eta}) - {experiment_name}",
        f"{experiment_name}_trajectory.png",
    )

    return trajectory, loss_values


# Example usage:
run_experiment(A, B, C, ETA_CONVERGE, experiment_name="converging_case")
run_experiment(A, B, C, ETA_DIVERGE, experiment_name="diverging_case")
