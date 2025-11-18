# Import required libraries for data processing, network analysis, and file handling
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

    # Step 1: Read JSON and merge node indices from the first 'size' solution groups, removing duplicates
    with open(json_path, "r", encoding="utf-8") as f:
        all_solutions = json.load(f)

    if size > len(all_solutions):
        raise ValueError(f"size={size} exceeds total number of solutions in JSON: {len(all_solutions)}")

    # Collect all unique node indices from the first 'size' solution groups
    combined_indices = set()
    for group in all_solutions[:size]:
        combined_indices.update(group)

    combined_indices = sorted(combined_indices)  # Fixed order for reproducibility

    # Step 2: Extract coordinates for these nodes from the CSV file
    df = pd.read_csv(csv_path)
    subset = df[df["node_index"].isin(combined_indices)].copy()
    subset = subset.sort_values("node_index").reset_index(drop=True)

    # Step 3: Load the road network and convert to undirected graph (ignore one-way street directions)
    G = ox.load_graphml(graphml_path)
    G = G.to_undirected()

    # Step 4: Match coordinates to nearest road network nodes using OSMnx distance utility
    lons = subset["lon"].values
    lats = subset["lat"].values
    road_nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)

    # Step 5: Build the distance matrix using road network shortest path lengths
    n = len(road_nodes)
    dist_matrix = np.zeros((n, n), dtype=float)

    # Compute shortest paths between all pairs of nodes
    for i in range(n):
        for j in range(i + 1, n):
            try:
                # Calculate shortest path distance on the road network (weight="length" for distance in meters)
                dist = nx.shortest_path_length(G, road_nodes[i], road_nodes[j], weight="length")
            except nx.NetworkXNoPath:
                # If no path exists between nodes, set distance to infinity
                dist = np.inf
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric matrix

    # Optional: Save distance matrix and metadata to disk for later use
    if out_matrix_path is not None:
        out_matrix_path = Path(out_matrix_path)
        out_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_matrix_path, dist_matrix)

    if out_meta_path is not None:
        out_meta_path = Path(out_meta_path)
        out_meta_path.parent.mkdir(parents=True, exist_ok=True)
        # Store metadata: size, input file paths, node indices, and corresponding road network node IDs
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
    # Configuration: Change 'size' value to combine different numbers of solution groups
    # Note: size cannot exceed 20 (the total number of groups in the JSON file)
    size = 10

    # Output file paths - can be customized as needed
    matrix_out = f"outputs/data/distance_matrix_size{size}.npy"
    meta_out = f"outputs/data/distance_matrix_size{size}_meta.json"

    # Build the combined distance matrix by calling the main function
    #！！！ attention: the name of the json file should be consistent with that in 3_generate_candidate_indices.py  !!!
    dist_matrix, subset_points, road_nodes = build_combined_distance_matrix(
        size=size,
        json_path="outputs/data/random_solutions_prepared.json",
        csv_path="outputs/data/marseille_1er_graph_nodes_projected.csv",
        graphml_path="outputs/maps/marseille_1er_road_network.graphml",
        out_matrix_path=matrix_out,
        out_meta_path=meta_out,
    )

    # Print summary information about the generated distance matrix
    print(f"Successfully generated and saved combined distance matrix for size={size}:")
    print("Matrix shape:", dist_matrix.shape)
    print("Matrix file:", matrix_out)
    print("Metadata file:", meta_out)
