import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox


def build_combined_distance_matrix(
    size: int,
    json_path: str = "outputs/data/random_solutions_prepared.json",
    csv_path: str = "outputs/data/marseille_1er_graph_nodes_projected.csv",
    graphml_path: str = "outputs/maps/marseille_1er_road_network.graphml",
    out_matrix_path: str | None = None,
    out_meta_path: str | None = None,
):
    """
    从 random_solutions_prepared.json 里取前 size 组解，把这些 node_index 合并去重，
    在道路网络上生成一个“总的”距离矩阵，并可选择保存到磁盘。

    参数
    ----
    size : int
        前几组解的集合，例如 size=10 就是把第 0~9 组的点合在一起。
    json_path : str
        备选点集合的 json 文件路径。
    csv_path : str
        候选节点坐标表（包含 node_index, lon, lat 等）。
    graphml_path : str
        道路网络图（OSMnx 导出的 .graphml）。
    out_matrix_path : str | None
        如果不为 None，则把距离矩阵保存为 .npy。
    out_meta_path : str | None
        如果不为 None，则保存一份 meta 信息（node_index 顺序等）为 .json。

    返回
    ----
    dist_matrix : np.ndarray
        n x n 的对称距离矩阵（道路最短路长度，单位：米）。
    subset : pd.DataFrame
        合并后的点在 CSV 中的记录（按 node_index 排好序）。
    road_nodes : list
        每个点匹配到的道路图节点 ID（与 dist_matrix 行列一一对应）。
    """

    # Step 1: 读取 json，合并前 size 组的 node_index，并去重
    with open(json_path, "r", encoding="utf-8") as f:
        all_solutions = json.load(f)

    if size > len(all_solutions):
        raise ValueError(f"size={size} 超过 json 中解的总数 {len(all_solutions)}")

    combined_indices = set()
    for group in all_solutions[:size]:
        combined_indices.update(group)

    combined_indices = sorted(combined_indices)  # 固定顺序，方便复现

    # Step 2: 从 CSV 中筛选出这些点的经纬度
    df = pd.read_csv(csv_path)
    subset = df[df["node_index"].isin(combined_indices)].copy()
    subset = subset.sort_values("node_index").reset_index(drop=True)

    # Step 3: 加载道路网络，并转为无向图（忽略单行道方向）
    G = ox.load_graphml(graphml_path)
    G = G.to_undirected()

    # Step 4: 利用经纬度匹配到最近的道路节点
    lons = subset["lon"].values
    lats = subset["lat"].values
    road_nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)

    # Step 5: 构建距离矩阵（道路最短路径长度）
    n = len(road_nodes)
    dist_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                dist = nx.shortest_path_length(G, road_nodes[i], road_nodes[j], weight="length")
            except nx.NetworkXNoPath:
                dist = np.inf
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # 可选：保存到磁盘
    if out_matrix_path is not None:
        out_matrix_path = Path(out_matrix_path)
        out_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_matrix_path, dist_matrix)

    if out_meta_path is not None:
        out_meta_path = Path(out_meta_path)
        out_meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "size": size,
            "json_path": str(json_path),
            "csv_path": str(csv_path),
            "graphml_path": str(graphml_path),
            "node_indices": subset["node_index"].tolist(),
            "road_nodes": [int(n) for n in road_nodes],
        }
        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return dist_matrix, subset, list(road_nodes)


if __name__ == "__main__":
    # 你可以在这里改 size，比如 10
    size = 10

    # 输出路径你也可以自己改名
    matrix_out = f"outputs/data/distance_matrix_size{size}.npy"
    meta_out = f"outputs/data/distance_matrix_size{size}_meta.json"

    dist_matrix, subset_points, road_nodes = build_combined_distance_matrix(
        size=size,
        json_path="outputs/data/random_solutions_prepared.json",
        csv_path="outputs/data/marseille_1er_graph_nodes_projected.csv",
        graphml_path="outputs/maps/marseille_1er_road_network.graphml",
        out_matrix_path=matrix_out,
        out_meta_path=meta_out,
    )

    print(f"完成：size={size} 的合并距离矩阵已生成并保存：")
    print("矩阵形状:", dist_matrix.shape)
    print("矩阵文件:", matrix_out)
    print("meta 文件:", meta_out)
