import numpy as np
import matplotlib.pyplot as plt

# Objective function
def f(x):
    return np.sum((x - 1) ** 2)

def cuckoo_search(f, n=100, D=10, lb=-5, ub=5, pa=0.25, alpha=0.3, iters=50):
    """
    Simple Cuckoo Search (updated implementation but same signature)
    - levy-like gaussian step
    - discovery: replace fraction pa of worst nests with new random nests
    - proper best / fitness updates
    """
    rnd = np.random.RandomState()  # use numpy RNG (no explicit seed in signature)
    X = rnd.uniform(lb, ub, (n, D))
    F = np.array([f(x) for x in X])
    best_idx = np.argmin(F)
    x_best, f_best = X[best_idx].copy(), F[best_idx]
    history = [f_best]

    for _ in range(iters):
        # generate new solutions and try to replace random nests
        for i in range(n):
            step = alpha * rnd.normal(0, 1, D)  # levy-like gaussian step
            X_new = np.clip(X[i] + step * (X[i] - x_best), lb, ub)
            f_new = f(X_new)

            j = rnd.randint(0, n)
            if f_new < F[j]:
                X[j], F[j] = X_new, f_new
                if f_new < f_best:
                    x_best, f_best = X_new.copy(), f_new

        # Discovery: replace a fraction pa of worst nests with new random solutions
        if pa > 0:
            num_replace = int(np.ceil(pa * n))
            if num_replace > 0:
                worst_idx = np.argsort(F)[-num_replace:]
                # create new nests by perturbing existing nests
                idx = rnd.randint(0, n, size=num_replace)
                k = rnd.randint(0, n, size=num_replace)
                r = rnd.uniform(0, 1, size=(num_replace, 1))  # scalar per nest, broadcast over D
                new_nests = np.clip(X[idx] + r * (X[idx] - X[k]), lb, ub)
                X[worst_idx] = new_nests
                F[worst_idx] = np.array([f(ind) for ind in new_nests])

                # update global best if needed
                cur_best_idx = np.argmin(F)
                if F[cur_best_idx] < f_best:
                    f_best = F[cur_best_idx]
                    x_best = X[cur_best_idx].copy()

        # ensure best is up-to-date
        cur_best_idx = np.argmin(F)
        if F[cur_best_idx] < f_best:
            f_best = F[cur_best_idx]
            x_best = X[cur_best_idx].copy()

        history.append(f_best)

    return x_best, f_best, history


# Run one case
x_best, f_best, history = cuckoo_search(f, iters=500)

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
