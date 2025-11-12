"""
Comparative experiment: Hill Climbing, Genetic Algorithm, Cuckoo Search
on Sphere and Ackley functions.

- Multi-run experiment to evaluate convergence speed, runtime, robustness.
- Produces:
    * average convergence curves (with percentiles)
    * boxplots of final best values across runs
    * printed runtime & simple stats

Requirements:
    pip install numpy matplotlib seaborn
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# -------------------------
# Benchmark functions
# -------------------------
def sphere(x):
    """Sphere function: global minimum at 0."""
    return np.sum(x ** 2)


def ackley(x):
    """Ackley function (standard form)."""
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = x.size
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.e


# -------------------------
# Optimizers
# -------------------------
def hill_climbing(func, D, lb, ub, iters=200, step_scale=0.1, seed=None):
    """
    Simple hill climbing (random neighbor, greedy).
    - start: random point in [lb, ub]
    - neighbor: gaussian perturbation scaled by step_scale*(ub-lb)
    - accept if better
    """
    rnd = np.random.RandomState(seed)
    x = rnd.uniform(lb, ub, D)
    fx = func(x)
    history = [fx]
    step_sigma = step_scale * (ub - lb)

    for _ in range(iters):
        cand = x + rnd.normal(0, step_sigma, D)
        cand = np.clip(cand, lb, ub)
        f_cand = func(cand)
        if f_cand < fx:
            x, fx = cand, f_cand
        history.append(fx)
    return x, fx, np.array(history)


def simulated_annealing(func, D, lb, ub, iters=500, T0=1.0, alpha=0.99, seed=None):
    """
    Simulated Annealing (SA) algorithm for function minimization.
    
    Parameters:
        func  : objective function to minimize
        D     : problem dimension
        lb, ub: lower and upper bounds (scalars or arrays)
        iters : number of iterations
        T0    : initial temperature
        alpha : cooling rate (temperature decay)
        seed  : random seed
    Returns:
        best_sol, best_fit, history
    """
    rnd = np.random.default_rng(seed)
    
    # Initialize solution randomly within bounds
    x = rnd.uniform(lb, ub, D)
    fx = func(x)
    
    # Record best solution
    best = x.copy()
    best_f = fx
    history = [fx]
    
    T = T0  # initial temperature

    for t in range(iters):
        # Generate new candidate by random perturbation
        x_new = x + rnd.normal(0, 0.1 * (ub - lb), D)
        x_new = np.clip(x_new, lb, ub)
        f_new = func(x_new)
        
        # Calculate energy difference
        delta = f_new - fx

        # Acceptance criterion:
        # Always accept better solution;
        # Accept worse solution with probability exp(-delta / T)
        if delta < 0 or rnd.random() < np.exp(-delta / T):
            x = x_new
            fx = f_new
        
        # Update best if necessary
        if fx < best_f:
            best = x.copy()
            best_f = fx
        
        # Cool down temperature
        T *= alpha

        history.append(best_f)
    
    return best, best_f, np.array(history)


def genetic_algorithm(func, D, lb, ub, iters=200, pop_size=30, cx_rate=0.6, mut_rate=0.2, mut_sigma=0.1, seed=None):
    """
    Real-valued GA:
    - population initialized uniformly
    - tournament selection (k=3)
    - uniform crossover
    - gaussian mutation
    """
    rnd = np.random.RandomState(seed)
    pop = rnd.uniform(lb, ub, (pop_size, D))
    fitness = np.array([func(ind) for ind in pop])
    history = [fitness.min()]

    def tournament_select():
        k = 3
        idx = rnd.choice(pop_size, k, replace=False)
        return pop[idx[np.argmin(fitness[idx])]].copy()

    for _ in range(iters):
        new_pop = []
        # elitism: keep best
        elite_idx = np.argmin(fitness)
        new_pop.append(pop[elite_idx].copy())

        while len(new_pop) < pop_size:
            # selection
            p1 = tournament_select()
            p2 = tournament_select()

            # crossover
            if rnd.rand() < cx_rate:
                mask = rnd.rand(D) < 0.5
                child = p1.copy()
                child[mask] = p2[mask]
            else:
                child = p1.copy()

            # mutation
            if rnd.rand() < mut_rate:
                child += rnd.normal(0, mut_sigma * (ub - lb), D)

            child = np.clip(child, lb, ub)
            new_pop.append(child)

        pop = np.array(new_pop)
        fitness = np.array([func(ind) for ind in pop])
        history.append(fitness.min())

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx], np.array(history)


def cuckoo_search(func, D, lb, ub, iters=200, n_nests=20, pa=0.25, alpha=0.5, seed=None):
    """
    Simplified Cuckoo Search (fixed):
    - levy-like gaussian step
    - discovery: replace fraction pa of worst nests with new random nests
    - proper best / fitness updates
    """
    rnd = np.random.RandomState(seed)
    nests = rnd.uniform(lb, ub, (n_nests, D))
    fitness = np.array([func(ind) for ind in nests])
    best_idx = np.argmin(fitness)
    best = nests[best_idx].copy()
    best_f = fitness[best_idx]
    history = [best_f]

    for _ in range(iters):
        # generate new solutions (one per nest) and try to replace random nests
        for i in range(n_nests):
            # levy-like step approximated by gaussian scaled by alpha
            step = alpha * rnd.normal(0, 1, D)
            new = nests[i] + step * (nests[i] - best)   # bias toward exploration around current nest and best
            new = np.clip(new, lb, ub)
            f_new = func(new)

            # compare with a random nest j
            j = rnd.randint(0, n_nests)
            if f_new < fitness[j]:
                nests[j] = new
                fitness[j] = f_new
                if f_new < best_f:
                    best = new.copy()
                    best_f = f_new

        # Discovery: replace a fraction pa of worst nests with new random solutions
        if pa > 0:
            num_replace = int(np.ceil(pa * n_nests))
            if num_replace > 0:
                worst_idx = np.argsort(fitness)[-num_replace:]
                new_nests = rnd.uniform(lb, ub, (num_replace, D))
                nests[worst_idx] = new_nests
                fitness[worst_idx] = np.array([func(ind) for ind in new_nests])

                # update global best if needed
                cur_best_idx = np.argmin(fitness)
                if fitness[cur_best_idx] < best_f:
                    best_f = fitness[cur_best_idx]
                    best = nests[cur_best_idx].copy()

        # ensure best is up-to-date
        cur_best_idx = np.argmin(fitness)
        if fitness[cur_best_idx] < best_f:
            best_f = fitness[cur_best_idx]
            best = nests[cur_best_idx].copy()

        history.append(best_f)

    return best, best_f, np.array(history)

# -------------------------
# Experiment driver
# -------------------------
def run_experiment(func, func_name, D=10, lb=-5, ub=5, runs=20, iters=200):
    """
    Run multiple independent runs for each algorithm and collect:
    - histories: (runs, iters+1)
    - final best values
    - runtimes
    """
    algos = {
        'HillClimb': lambda seed: hill_climbing(func, D, lb, ub, iters=iters, step_scale=0.05, seed=seed),
        'SA': lambda seed: simulated_annealing(func, D, lb, ub, iters=500, T0=1.0, alpha=0.99, seed=seed),
        'GA': lambda seed: genetic_algorithm(func, D, lb, ub, iters=iters, pop_size=30, cx_rate=0.7, mut_rate=0.3, mut_sigma=0.05, seed=seed),
        'Cuckoo': lambda seed: cuckoo_search(func, D, lb, ub, iters=iters, n_nests=40, pa=0.25, alpha=0.3, seed=seed),
    
    }

    results = {}
    for name, alg in algos.items():
        histories = []
        finals = []
        times = []
        print(f"Running {name} on {func_name} (D={D}), runs={runs} ...")
        for run in range(runs):
            seed = int(time.time() * 1000) % 2**31  # varying seed
            t0 = time.time()
            _, fbest, history = alg(seed)
            t1 = time.time()
            histories.append(history)
            finals.append(fbest)
            times.append(t1 - t0)
        histories = np.array(histories)  # shape (runs, iters+1)
        results[name] = {
            'histories': histories,
            'finals': np.array(finals),
            'times': np.array(times)
        }
        print(f"  avg final: {np.mean(finals):.4e}, median: {np.median(finals):.4e}, avg time: {np.mean(times):.4f}s")
    return results


# -------------------------
# Plotting helpers
# -------------------------
def plot_convergence(results, iters, title):
    plt.figure(figsize=(8, 5))
    for name, data in results.items():
        histories = data['histories'][:, :iters+1]
        mean = np.mean(histories, axis=0)
        p25 = np.percentile(histories, 25, axis=0)
        p75 = np.percentile(histories, 75, axis=0)
        plt.plot(mean, label=f"{name} mean")
        plt.fill_between(range(iters+1), p25, p75, alpha=0.2)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Best objective (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_boxplots(results, title):
    plt.figure(figsize=(8, 5))
    data = [results[name]['finals'] for name in results.keys()]
    sns.boxplot(data=data)
    plt.xticks(range(len(results)), list(results.keys()))
    plt.yscale('log')
    plt.ylabel("Final best objective (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -------------------------
# Main experiment
# -------------------------
if __name__ == "__main__":
    # experiment configuration
    runs = 20
    iters = 400
    dims = [2, 8, 32]  # change / expand to test scalability

    # For Ackley typical domain is [-32, 32]; Sphere often [-5,5]
    for D in dims:
        print("\n" + "="*60)
        print(f"Dimension = {D}")
        print("="*60 + "\n")

        # Sphere experiment
        sphere_results = run_experiment(sphere, "Sphere", D=D, lb=-5, ub=5, runs=runs, iters=iters)
        plot_convergence(sphere_results, iters, title=f"Convergence on Sphere (D={D})")
        plot_boxplots(sphere_results, title=f"Final best on Sphere (D={D})")

        # Ackley experiment
        ack_results = run_experiment(ackley, "Ackley", D=D, lb=-32, ub=32, runs=runs, iters=iters)
        plot_convergence(ack_results, iters, title=f"Convergence on Ackley (D={D})")
        plot_boxplots(ack_results, title=f"Final best on Ackley (D={D})")

        # Print runtime summary
        print("Runtime summary (avg seconds):")
        for name in sphere_results.keys():
            print(f"  {name} (Sphere): {sphere_results[name]['times'].mean():.4f}s, (Ackley): {ack_results[name]['times'].mean():.4f}s")

    print("\nExperiment finished.")
