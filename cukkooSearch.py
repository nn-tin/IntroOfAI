import numpy as np
import matplotlib.pyplot as plt

# Objective function
def f(x):
    return np.sum((x - 1) ** 2)

import numpy as np

def cuckoo_search(f, n=3, D=5, lb=-5, ub=5, pa=0.25, alpha=0.3, iters=50):
    """
    Simple Cuckoo Search with local random walk applied to each nest individually.
    """
    X = np.random.uniform(lb, ub, (n, D))
    F = np.array([f(x) for x in X])
    best_idx = np.argmin(F)
    x_best, f_best = X[best_idx].copy(), F[best_idx]
    history = [f_best]

    for _ in range(iters):
        # Lévy flight (global search)
        i = np.random.randint(n)
        j = np.random.randint(n)

        levy_step = alpha * np.random.randn(D)
        X_new = np.clip(X[i] + levy_step, lb, ub)
        f_new = f(X_new)
        if f_new < F[j]:
            X[j], F[j] = X_new, f_new
        if f_new < f_best:
            x_best, f_best = X_new.copy(), f_new

        # Local random walk (exploration) over each nest
        for idx in range(n):
            if np.random.rand() < pa:
                others = [x for x in range(n) if x != idx]
                k = np.random.choice(others)
                r = np.random.rand()
                X_new = np.clip(X[idx] + r * (X[idx] - X[k]), lb, ub)
                f_new = f(X_new)
                if f_new < F[idx]:
                    X[idx], F[idx] = X_new, f_new
                if f_new < f_best:
                    x_best, f_best = X_new.copy(), f_new

        history.append(f_best)

    return x_best, f_best, history


# Run one case
x_best, f_best, history = cuckoo_search(f, iters=200)

# Convergence curve
plt.figure(figsize=(6, 4))
plt.plot(history, color='royalblue', marker='o', markersize=1, linewidth=0.5)
plt.title("Convergence Curve of Cuckoo Search")
plt.xlabel("Iteration")
plt.ylabel("Best Objective Value")
plt.grid(True)
plt.show()


print("=== Final Result ===")
print(f"x_best = {np.round(x_best, 3)}")
print(f"f_best = {f_best:.6f}")
