import numpy as np
import matplotlib.pyplot as plt

# Objective function
def f(x):
    return np.sum((x - 1) ** 2)

# Cuckoo Search visualization (for D=2)
def cuckoo_search_2D(n=3, lb=-5, ub=5, pa=0.25, alpha=0.2, iters=30):
    D = 2
    X = np.random.uniform(lb, ub, (n, D))
    F = np.array([f(x) for x in X])
    best_idx = np.argmin(F)
    x_best, f_best = X[best_idx].copy(), F[best_idx]

    history_best = [f_best]
    paths = [[x.copy()] for x in X]  # track paths of each cuckoo

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

        history_best.append(f_best)
        for p, pos in zip(paths, X):
            p.append(pos.copy())

    return x_best, f_best, history_best, paths

# Run
x_best, f_best, history, paths = cuckoo_search_2D(iters=200)

# 3D Surface Plot + Paths
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
Xg = np.linspace(-5, 5, 100)
Yg = np.linspace(-5, 5, 100)
Xg, Yg = np.meshgrid(Xg, Yg)
Zg = (Xg - 1)**2 + (Yg - 1)**2
ax.plot_surface(Xg, Yg, Zg, cmap='viridis', alpha=0.6)

# Plot cuckoo trajectories
for i, path in enumerate(paths):
    path = np.array(path)
    Z = np.array([f(x) for x in path])
    ax.plot(path[:, 0], path[:, 1], Z, marker='o', markersize=3, label=f'Cuckoo {i+1}')

# Mark best point
ax.scatter(x_best[0], x_best[1], f_best, color='red', s=80, label='Best found')
ax.set_title("Cuckoo Search Path on Objective Surface (2D)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x1, x2)")
ax.legend()
plt.show()

# Convergence curve
plt.figure(figsize=(6, 4))
plt.plot(history, color='royalblue', marker='o', linewidth=1.5)
plt.title("Convergence Curve of Cuckoo Search")
plt.xlabel("Iteration")
plt.ylabel("Best Objective Value")
plt.grid(True)
plt.show()

print("=== Final Result ===")
print(f"x_best = {np.round(x_best, 3)}")
print(f"f_best = {f_best:.6f}")
