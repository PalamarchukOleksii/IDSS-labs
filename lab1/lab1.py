import sympy as sp
import numpy as np

A = 0.4
B = 0.8
C = 0.5
ETA = 0.1

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
    
    for t in range(T):
        grad_val = np.array(grad_func(w_num[0], w_num[1], a_val, b_val, c_val)).flatten()
        step = eta * grad_val
        w_num = w_num - step
        
        if t % 10 == 0:
            print(f"Iteration {t}: w = [{w_num[0]:.20f}, {w_num[1]:.20f}]")
        
        if tol is not None and np.linalg.norm(step) < tol:
            break
    
    print(f"Final w* on iter {t}: [{w_num[0]:.20f}, {w_num[1]:.20f}]")
    return w_num

w_opt = gradient_descent(A, B, C, ETA, tol=0.0001)
