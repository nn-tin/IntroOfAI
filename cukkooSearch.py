import numpy as np
import matplotlib.pyplot as plt

# Objective function
def f(x):
    return np.sum((x - 1) ** 2)

# Simple Cuckoo Search (one run)
def cuckoo_search(n=3, D=5, lb=-5, ub=5, pa=0.25, alpha=0.3, iters=50):
    X = np.random.uniform(lb, ub, (n, D))
    F = np.array([f(x) for x in X])
    best_idx = np.argmin(F)
    x_best, f_best = X[best_idx].copy(), F[best_idx]
    history = [f_best]

    for _ in range(iters):
        # Lévy flight (global search)
        i = np.random.randint(n)
        levy_step = alpha * np.random.randn(D)
        X_new = np.clip(X[i] + levy_step, lb, ub)
        f_new = f(X_new)
        if f_new < F[i]:
            X[i], F[i] = X_new, f_new
        if f_new < f_best:
            x_best, f_best = X_new.copy(), f_new

        # Local random walk (exploration)
        j, k = np.random.choice(n, 2, replace=False)
        if np.random.rand() < pa:
            r = np.random.rand()
            X_new = np.clip(X[j] + r * (X[j] - X[k]), lb, ub)
            f_new = f(X_new)
            if f_new < F[j]:
                X[j], F[j] = X_new, f_new
            if f_new < f_best:
                x_best, f_best = X_new.copy(), f_new

        history.append(f_best)

    return x_best, f_best, history

# Run one case
x_best, f_best, history = cuckoo_search(iters=500)

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
