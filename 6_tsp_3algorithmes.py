"""
TSP comparison script: Greedy, Monte Carlo, and NetworkX TSP
Keeps bilingual comments from the original script (Chinese + English).
Saves results JSON and a PNG plot into the data folder.

Save as: tsp_greedy_monte_networkx_compare.py
Run: python tsp_greedy_monte_networkx_compare.py
"""

import json
from pathlib import Path
import random
import math

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem


# ----------------------------
# Helper: Greedy TSP (Nearest Neighbor)
# ----------------------------
def greedy_tsp_from_matrix(dist_matrix):
    """
    Greedy nearest-neighbor TSP on a full distance matrix.
    Returns (total_length, tour_list) where tour_list is closed (last == first).
    """
    n = dist_matrix.shape[0]
    if n == 0:
        return 0.0, []
    visited = set([0])
    tour = [0]
    current = 0
    total = 0.0

    while len(visited) < n:
        best_j = None
        best_d = float("inf")
        for j in range(n):
            if j in visited:
                continue
            d = float(dist_matrix[current, j])
            if math.isfinite(d) and d < best_d:
                best_d = d
                best_j = j
        if best_j is None:
            # No reachable unvisited node (disconnected) — break
            break
        visited.add(best_j)
        tour.append(best_j)
        total += best_d
        current = best_j

    # close the loop if we visited full set and closing edge finite
    if len(tour) == n and math.isfinite(float(dist_matrix[current, tour[0]])):
        total += float(dist_matrix[current, tour[0]])
        tour.append(tour[0])

    return float(total), tour


# ----------------------------
# Helper: Monte Carlo TSP (random permutations sampling)
# ----------------------------
def monte_carlo_tsp_from_matrix(dist_matrix, samples=3000, seed=None):
    """
    Monte Carlo random-sampling TSP. Draws `samples` random permutations
    and returns the best found: (best_length, best_tour_closed)
    """
    if seed is not None:
        random.seed(seed)

    n = dist_matrix.shape[0]
    if n == 0:
        return 0.0, []

    nodes = list(range(n))
    best_len = float("inf")
    best_tour = None

    for _ in range(samples):
        perm = random.sample(nodes, n)
        total = 0.0
        valid = True
        for i in range(n - 1):
            d = float(dist_matrix[perm[i], perm[i + 1]])
            if not math.isfinite(d):
                valid = False
                break
            total += d
        # close loop
        if not valid:
            continue
        last_edge = float(dist_matrix[perm[-1], perm[0]])
        if not math.isfinite(last_edge):
            continue
        total += last_edge

        if total < best_len:
            best_len = total
            best_tour = perm + [perm[0]]

    if best_tour is None:
        # no valid permutation found
        return float("inf"), []
    return float(best_len), best_tour


# ----------------------------
# Helper: NetworkX TSP (default approximation)
# ----------------------------
def networkx_tsp_from_matrix(dist_matrix):
    """
    Build a NetworkX graph from the distance matrix and use
    traveling_salesman_problem(H, cycle=True, weight="weight").
    Returns (length, tour_closed) where tour indices are matrix indices.
    """
    n = dist_matrix.shape[0]
    H = nx.Graph()
    for i in range(n):
        H.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            w = float(dist_matrix[i, j])
            if math.isfinite(w):
                H.add_edge(i, j, weight=w)
    # Compute approximate TSP tour
    tour = traveling_salesman_problem(H, cycle=True, weight="weight")
    # compute length using matrix (safer if matrix has small asymmetries)
    total = 0.0
    for k in range(len(tour)-1):
        total += float(dist_matrix[tour[k], tour[k+1]])
    return float(total), list(tour)


# ----------------------------
# Main function: mimics structure of your original script, keeps comments
# ----------------------------
def compute_tsp_all_methods(matrix_path: str, meta_path: str, samples=3000, seed=None):
    """
    从磁盘读取距离矩阵和 meta 信息，构建完全图，求 TSP 近似回路（Greedy, Monte Carlo, NetworkX）。
    Read the distance matrix and meta information from disk and compute greedy, monte-carlo and networkx TSP.
    Keeps the mapping from matrix indices to node_indices as in original script.
    """
    matrix_path = Path(matrix_path)
    meta_path = Path(meta_path)

    dist_matrix = np.load(matrix_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    node_indices = meta.get("node_indices", None)
    n = dist_matrix.shape[0]

    # Validate shapes
    if node_indices is None or len(node_indices) != n:
        print(f"Warning: meta 'node_indices' missing or length mismatch for {matrix_path}.")
        node_indices = list(range(n))

    # Greedy
    g_length, g_tour = greedy_tsp_from_matrix(dist_matrix)

    # Monte Carlo
    mc_length, mc_tour = monte_carlo_tsp_from_matrix(dist_matrix, samples=samples, seed=seed)

    # NetworkX
    nx_length, nx_tour = networkx_tsp_from_matrix(dist_matrix)

    # Map visiting tour indices back to node_indices if available
    visiting_node_indices_greedy = [node_indices[i] for i in g_tour] if g_tour else []
    visiting_node_indices_monte = [node_indices[i] for i in mc_tour] if mc_tour else []
    visiting_node_indices_nx = [node_indices[i] for i in nx_tour] if nx_tour else []

    return {
        "matrix_path": str(matrix_path),
        "meta_path": str(meta_path),
        "n": n,
        "greedy_length": float(g_length),
        "greedy_tour_matrix_idx": g_tour,
        "greedy_tour_node_idx": visiting_node_indices_greedy,
        "monte_length": float(mc_length) if math.isfinite(mc_length) else None,
        "monte_tour_matrix_idx": mc_tour,
        "monte_tour_node_idx": visiting_node_indices_monte,
        "networkx_length": float(nx_length),
        "networkx_tour_matrix_idx": nx_tour,
        "networkx_tour_node_idx": visiting_node_indices_nx,
        "meta": meta,
    }


# ----------------------------
# Execution entrypoint
# ----------------------------
if __name__ == "__main__":
    # 这里的 size 要和你生成矩阵时保持一致
    # The sizes / filenames here must match the matrices you have.
    sizes = [1, 4, 7, 10]

    # Folder where your .npy and .json files live
    # 使用相对路径（以脚本所在的项目根目录为基准）
    PROJECT_DIR = Path(__file__).resolve().parent
    base_folder = PROJECT_DIR / "outputs" / "data"

    # Monte Carlo sampling config
    monte_samples = 5000
    monte_seed = 42  # optional reproducibility; set to None to randomize differently each run

    # Collect results
    results = {}
    greedy_values = []
    monte_values = []
    networkx_values = []
    labels = []

    for size in sizes:
        matrix_path = base_folder / f"distance_matrix_size{size}.npy"
        meta_path = base_folder / f"distance_matrix_size{size}_meta.json"

        if not matrix_path.exists():
            print(f"ERROR: Missing matrix file: {matrix_path}")
            continue
        if not meta_path.exists():
            print(f"ERROR: Missing meta file: {meta_path}")
            continue

        print(f"Processing size {size} -> {matrix_path.name}")
        out = compute_tsp_all_methods(str(matrix_path), str(meta_path),
                                     samples=monte_samples, seed=monte_seed)

        results[f"size_{size}"] = out
        greedy_values.append(out["greedy_length"])
        monte_values.append(out["monte_length"] if out["monte_length"] is not None else float("nan"))
        networkx_values.append(out["networkx_length"])
        labels.append(f"size{size}")

        print(f"  Greedy length:    {out['greedy_length']:.6f}")
        if out["monte_length"] is None:
            print("  Monte Carlo:      no valid tour found")
        else:
            print(f"  Monte length:     {out['monte_length']:.6f}")
        print(f"  NetworkX length:  {out['networkx_length']:.6f}")

    # ----------------------------
    # Save results JSON
    # ----------------------------
    json_out_path = base_folder / "tsp_greedy_monte_networkx_results.json"
    with open(json_out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON results to: {json_out_path}")

    # ----------------------------
    # Plot comparison: Greedy vs Monte Carlo vs NetworkX
    # ----------------------------
    plt.figure(figsize=(8, 5))
    # plot Greedy (blue)
    plt.plot(labels, greedy_values, marker="o", linestyle="-", color="tab:blue", label="Greedy (Nearest Neighbor)")
    # plot Monte Carlo (orange)
    plt.plot(labels, monte_values, marker="o", linestyle="--", color="tab:orange", label=f"Monte Carlo (samples={monte_samples})")
    # plot NetworkX (green)
    plt.plot(labels, networkx_values, marker="o", linestyle="-.", color="tab:green", label="NetworkX TSP (approx)")

    plt.xlabel("Matrix (by size)")
    plt.ylabel("TSP Tour Length")
    # Title without the word "graph"
    plt.title("TSP: Greedy vs Monte Carlo vs NetworkX")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    png_out_path = base_folder / "tsp_greedy_monte_networkx_compare.png"
    plt.savefig(png_out_path, dpi=300)
    plt.show()

    print(f"Saved plot to: {png_out_path}")