import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
sns.set(style="whitegrid")

# -------------------------
# Graph utilities
# -------------------------
def random_graph(n_nodes, edge_prob, seed=None):
    rnd = np.random.RandomState(seed)
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    edges = list(G.edges())
    return G, edges

def fitness_conflicts(colors, edges):
    """Count number of conflicting edges (same color on both ends)."""
    return sum(colors[u] == colors[v] for u, v in edges)

# -------------------------
# Hill Climbing
# -------------------------
def hill_climbing(G, edges, k_colors=4, iters=500, seed=None):
    rnd = np.random.RandomState(seed)
    n = len(G)
    colors = rnd.randint(0, k_colors, n)
    f = fitness_conflicts(colors, edges)
    history = [f]
    for _ in range(iters):
        v = rnd.randint(0, n)
        old_color = colors[v]
        new_color = rnd.randint(0, k_colors)
        colors[v] = new_color
        f_new = fitness_conflicts(colors, edges)
        if f_new <= f:
            f = f_new
        else:
            colors[v] = old_color  # reject
        history.append(f)
    return colors, f, np.array(history)

# -------------------------
# Genetic Algorithm
# -------------------------
def genetic_algorithm(G, edges, k_colors=4, pop_size=30, iters=300, cx_rate=0.7, mut_rate=0.2, seed=None):
    rnd = np.random.RandomState(seed)
    n = len(G)
    pop = rnd.randint(0, k_colors, (pop_size, n))
    fitness = np.array([fitness_conflicts(ind, edges) for ind in pop])
    history = [fitness.min()]

    def tournament_select():
        idx = rnd.choice(pop_size, 3, replace=False)
        return pop[idx[np.argmin(fitness[idx])]].copy()

    for _ in range(iters):
        new_pop = []
        elite = pop[np.argmin(fitness)].copy()
        new_pop.append(elite)
        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            child = p1.copy()
            # crossover
            if rnd.rand() < cx_rate:
                mask = rnd.rand(n) < 0.5
                child[mask] = p2[mask]
            # mutation
            if rnd.rand() < mut_rate:
                i = rnd.randint(0, n)
                child[i] = rnd.randint(0, k_colors)
            new_pop.append(child)
        pop = np.array(new_pop)
        fitness = np.array([fitness_conflicts(ind, edges) for ind in pop])
        history.append(fitness.min())
    best = pop[np.argmin(fitness)].copy()
    return best, fitness.min(), np.array(history)

# -------------------------
# Discrete Cuckoo Search
# -------------------------
def cuckoo_search(G, edges, k_colors=4, n_nests=20, iters=300, pa=0.25, seed=None):
    rnd = np.random.RandomState(seed)
    n = len(G)
    nests = rnd.randint(0, k_colors, (n_nests, n))
    fitness = np.array([fitness_conflicts(ind, edges) for ind in nests])
    best = nests[np.argmin(fitness)].copy()
    best_f = fitness.min()
    history = [best_f]

    for _ in range(iters):
        for i in range(n_nests):
            j = rnd.randint(0, n_nests)
            if rnd.rand() < 0.5:
                # swap colors of two vertices (discrete perturbation)
                a, b = rnd.randint(0, n, 2)
                new = nests[i].copy()
                new[a], new[b] = new[b], new[a]
            else:
                # random recolor
                new = nests[i].copy()
                idx = rnd.randint(0, n)
                new[idx] = rnd.randint(0, k_colors)
            f_new = fitness_conflicts(new, edges)
            if f_new < fitness[j]:
                nests[j] = new
                fitness[j] = f_new
                if f_new < best_f:
                    best_f = f_new
                    best = new.copy()
        # discovery
        K = int(pa * n_nests)
        worst = np.argsort(fitness)[-K:]
        nests[worst] = rnd.randint(0, k_colors, (K, n))
        fitness[worst] = [fitness_conflicts(ind, edges) for ind in nests[worst]]
        if fitness.min() < best_f:
            best = nests[np.argmin(fitness)].copy()
            best_f = fitness.min()
        history.append(best_f)
    return best, best_f, np.array(history)

# -------------------------
# Simulated Annealing
# -------------------------
def simulated_annealing(G, edges, k_colors=4, iters=500, T0=1.0, alpha=0.99, seed=None):
    rnd = np.random.RandomState(seed)
    n = len(G)
    colors = rnd.randint(0, k_colors, n)
    f = fitness_conflicts(colors, edges)
    best = colors.copy()
    best_f = f
    history = [f]
    T = T0

    for _ in range(iters):
        v = rnd.randint(0, n)
        old_color = colors[v]
        new_color = rnd.randint(0, k_colors)
        colors[v] = new_color
        f_new = fitness_conflicts(colors, edges)
        delta = f_new - f
        if delta < 0 or rnd.rand() < np.exp(-delta / T):
            f = f_new
            if f_new < best_f:
                best = colors.copy()
                best_f = f_new
        else:
            colors[v] = old_color  # reject
        T *= alpha
        history.append(best_f)
    return best, best_f, np.array(history)

# -------------------------
# Run experiment
# -------------------------
def run_experiment():
    graph_configs = [
        {'n_nodes': 5, 'edge_prob': 0.15, 'k_colors': 3},
        {'n_nodes': 20, 'edge_prob': 0.1, 'k_colors': 4},
        {'n_nodes': 80, 'edge_prob': 0.08, 'k_colors': 6},
        {'n_nodes': 100, 'edge_prob': 0.08, 'k_colors': 6},

    ]
    runs = 5
    iters = 300
    algos = {
        "HillClimb": hill_climbing,
        "Genetic": genetic_algorithm,
        "Cuckoo": cuckoo_search,
        "SimAnneal": simulated_annealing
    }

    for config in graph_configs:
        print(f"\nGraph config: Nodes={config['n_nodes']}, Edge probability={config['edge_prob']}, Colors={config['k_colors']}")
        G, edges = random_graph(config['n_nodes'], config['edge_prob'], seed=42)
        results = {}
        for name, alg in algos.items():
            all_hist = []
            finals = []
            for run in range(runs):
                _, f, h = alg(G, edges, k_colors=config['k_colors'], iters=iters, seed=run)
                all_hist.append(h)
                finals.append(f)
            results[name] = {"hist": np.array(all_hist), "finals": np.array(finals)}

        # Plot convergence
        plt.figure(figsize=(8, 5))
        for name, data in results.items():
            mean = data["hist"].mean(axis=0)
            plt.plot(mean, label=name)
        plt.xlabel("Iteration")
        plt.ylabel("Conflicts")
        plt.title(f"Graph Coloring - average convergence (Nodes={config['n_nodes']})")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Boxplot final results
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=[results[n]["finals"] for n in algos.keys()])
        plt.xticks(range(len(algos)), list(algos.keys()))
        plt.ylabel("Final conflicts")
        plt.title(f"Final results (Nodes={config['n_nodes']}, lower is better)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_experiment()
