import json
from pathlib import Path

import numpy as np
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem


def compute_tsp_from_matrix(matrix_path: str, meta_path: str):
    """
    从磁盘读取距离矩阵和 meta 信息，构建完全图，求一个 TSP 近似回路。
    """

    matrix_path = Path(matrix_path)
    meta_path = Path(meta_path)

    dist_matrix = np.load(matrix_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    node_indices = meta["node_indices"]
    n = dist_matrix.shape[0]

    # 构建完全图（对 inf 的边可以选择不加边）
    H = nx.Graph()
    for i in range(n):
        H.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            w = float(dist_matrix[i, j])
            if np.isinf(w):
                # 如果真的存在不连通情况，这里可以选择跳过
                continue
            H.add_edge(i, j, weight=w)

    # 近似求解 TSP 回路
    tour = traveling_salesman_problem(H, cycle=True, weight="weight")

    # 计算总长度
    length = sum(dist_matrix[tour[k], tour[k + 1]] for k in range(len(tour) - 1))

    # 把访问顺序映射回 node_index
    visiting_node_indices = [node_indices[i] for i in tour]

    return float(length), tour, visiting_node_indices, meta


if __name__ == "__main__":
    # 这里的 size 要和你生成矩阵时保持一致
    size = 10

    matrix_path = f"outputs/data/distance_matrix_size{size}.npy"
    meta_path = f"outputs/data/distance_matrix_size{size}_meta.json"

    tsp_length, tour, visiting_node_indices, meta = compute_tsp_from_matrix(
        matrix_path, meta_path
    )

    print("=== TSP 结果 ===")
    print(f"组合 size = {meta['size']}")
    print(f"TSP 近似回路总长度 ≈ {tsp_length:.2f} 米")
    print("访问顺序（矩阵索引）:", tour)
    print("访问顺序对应的 node_index:", visiting_node_indices)
