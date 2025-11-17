# -*- coding: utf-8 -*-
"""
根据候选点与马赛第1区路网，构建 TSP 用“完全图”：
- 节点：候选点（带经纬度等属性）
- 边：任意两点之间都有一条边
- 边权重：真实道路网络上的最短路径长度（米），写在 weight / length 属性里
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox

# -----------------------
# 路径配置（按需修改）
# -----------------------
GRAPHML_DENSIFIED = "outputs/maps/marseille_1er_densified_network.graphml"
GRAPHML_RAW = "outputs/maps/marseille_1er_network.graphml"
SOL_JSON = "outputs/data/random_solutions_prepared.json"
CSV_PATTERN = "outputs/data/marseille_1er_graph_nodes_projected.csv"


# -----------------------
# 工具函数
# -----------------------

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return x


def _safe_int(x):
    try:
        # 允许 "4.0" 这类字符串转成 4
        return int(float(x))
    except Exception:
        return x


def pick_graph(graphml_main: str, graphml_fallback: str) -> nx.Graph:
    """
    优先加载插值后的图，失败再用原始图；
    并进行常见 dtype 修复，返回无向图。
    """
    path = graphml_main if os.path.exists(graphml_main) else graphml_fallback
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到图文件：\n- {graphml_main}\n- {graphml_fallback}")

    try:
        G = ox.load_graphml(path)
        G = G.to_undirected()
    except ValueError as e:
        print(f"[warn] ox.load_graphml 解析失败，将回退到 networkx.read_graphml 并进行 dtype 修复：{e}")
        G = nx.read_graphml(path)

        # 规范 node 属性的数值类型
        node_float_keys = ["x", "y", "lon", "lat"]
        node_int_keys_maybe = ["street_count"]

        for _, data in G.nodes(data=True):
            for k in node_float_keys:
                if k in data:
                    data[k] = _safe_float(data[k])
            for k in node_int_keys_maybe:
                if k in data:
                    data[k] = _safe_int(data[k])

        # 规范 edge 的长度字段
        edge_len_keys = ["length", "length_m", "distance", "weight"]
        for _, _, _, data in G.edges(keys=True, data=True):
            for k in edge_len_keys:
                if k in data:
                    data[k] = _safe_float(data[k])

        G = G.to_undirected()

    return G


def largest_undirected_component(G: nx.Graph) -> nx.Graph:
    """
    保留无向图的最大连通分量，丢弃其它孤立块。
    这样可以保证理论上任意两节点之间都有路可走。
    """
    if nx.is_connected(G):
        print("[Info] Graph is already connected.")
        return G
    comps = list(nx.connected_components(G))
    largest = max(comps, key=len)
    print(f"[Info] Graph has {len(comps)} components, keep largest with {len(largest)} nodes.")
    return G.subgraph(largest).copy()


def load_solution_indices(json_path: str, n: int = 1) -> list:
    """
    读取前 n 组解（默认第0组），合并为一个去重后的候选点编号列表（保持原顺序）。
    :param json_path: JSON 文件路径
    :param n: 要合并的解组数量（从第0组开始）
    :return: 合并并去重的编号列表
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到解文件: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        sols = json.load(f)

    if not sols:
        raise ValueError("解列表为空")

    if n > len(sols):
        n = len(sols)  # 防止超出范围

    def extract_indices(solution):
        if isinstance(solution, dict):
            for key in ("solution", "indices", "nodes", "ids"):
                if key in solution:
                    return list(solution[key])
            flat = []
            for v in solution.values():
                if isinstance(v, (list, tuple, set)):
                    flat.extend(v)
            if flat:
                return flat
            raise ValueError("无法从解中提取编号")
        # list / tuple / set 之类
        return list(solution)

    # 合并前 n 组解
    combined = []
    for i in range(n):
        combined.extend(extract_indices(sols[i]))

    # 去重但保持原顺序
    seen = set()
    unique_combined = [x for x in combined if not (x in seen or seen.add(x))]

    return unique_combined


def load_candidates_df(csv_pattern: str) -> pd.DataFrame:
    """读取 output/data 下的一张候选表；如多张，取第一张匹配含关键列的。"""
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        raise FileNotFoundError(f"未在 {csv_pattern} 匹配到任何 CSV")
    for p in csv_files:
        df = pd.read_csv(p)
        cols = set(c.lower() for c in df.columns)
        needed_any_id = {"candidate_id", "candidate_id_new", "id", "index", "node_index"}
        if cols & needed_any_id:
            return df
    return pd.read_csv(csv_files[0])


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    规范列名，返回含: cid, lon, lat, x, y, osmid（若有）
    这里假设你的 CSV 中有 node_index 作为候选点编号。
    """
    # 找编号列
    id_candidates = ["node_index", "candidate_id", "candidate_id_new", "id", "index"]
    id_col = None
    for c in id_candidates:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        # 尝试忽略大小写
        low = {c.lower(): c for c in df.columns}
        for c in id_candidates:
            if c in low:
                id_col = low[c]
                break
    if id_col is None:
        raise KeyError(f"未找到编号列，期望之一: {id_candidates}")

    # 经纬度/x/y 列
    lon_col = next((c for c in ["lon", "longitude", "x"] if c in df.columns), None)
    lat_col = next((c for c in ["lat", "latitude", "y"] if c in df.columns), None)

    x_col = next((c for c in ["x", "lon", "longitude"] if c in df.columns), None)
    y_col = next((c for c in ["y", "lat", "latitude"] if c in df.columns), None)

    osmid_col = next((c for c in ["osmid", "osmid_node", "node", "node_id"] if c in df.columns), None)

    rename_dict = {id_col: "cid"}
    if lon_col:
        rename_dict[lon_col] = "lon"
    if lat_col:
        rename_dict[lat_col] = "lat"
    if x_col:
        rename_dict[x_col] = "x"
    if y_col:
        rename_dict[y_col] = "y"
    if osmid_col:
        rename_dict[osmid_col] = "osmid"

    return df.rename(columns=rename_dict)


def filter_subset(df: pd.DataFrame, indices: list) -> pd.DataFrame:
    """按照解的编号过滤，并以编号排序。"""
    if "cid" not in df.columns:
        raise KeyError("内部错误：缺少标准化后的列 cid")
    sub = df[df["cid"].isin(indices)].copy()
    if sub.empty:
        raise ValueError("根据解的编号在CSV里没有匹配到任何行")
    sub.sort_values(by="cid", inplace=True)
    sub.reset_index(drop=True, inplace=True)
    return sub


def map_points_to_nodes(G: nx.Graph, sub: pd.DataFrame) -> np.ndarray:
    """
    把候选点映射到路网节点：
       1) 若有 osmid 且存在于图中，直接用；
       2) 否则优先用经纬度 lon/lat 最近节点；
       3) 若无经纬度则尝试 x/y（OSMnx 存的是 x=lon, y=lat）。
    """
    nodes_set = set(G.nodes)

    # 情况1：osmid
    if "osmid" in sub.columns and sub["osmid"].notna().any():
        node_ids = []
        for v in sub["osmid"].tolist():
            vv = v
            if isinstance(v, str) and "," in v:
                vv = v.split(",")[0].strip()
            try:
                if isinstance(vv, float) and vv.is_integer():
                    vv = int(vv)
                else:
                    vv = int(vv)
            except Exception:
                pass
            if vv in nodes_set:
                node_ids.append(vv)
            else:
                node_ids = None
                break
        if node_ids is not None:
            return np.array(node_ids)

    # 情况2：lon/lat
    if {"lon", "lat"}.issubset(sub.columns) and sub["lon"].notna().all() and sub["lat"].notna().all():
        # use positional args and ensure float numpy arrays to satisfy signature
        return ox.distance.nearest_nodes(G, sub["lon"].astype(float).to_numpy(), sub["lat"].astype(float).to_numpy())

    # 情况3：x/y -> 作为经纬度使用
    if {"x", "y"}.issubset(sub.columns) and sub["x"].notna().all() and sub["y"].notna().all():
        # use positional args and ensure float numpy arrays to satisfy signature
        return ox.distance.nearest_nodes(G, sub["x"].astype(float).to_numpy(), sub["y"].astype(float).to_numpy())

    raise ValueError("无法将点映射到路网节点：缺少 osmid 或 lon/lat 或 x/y 信息")


def edge_weight_key(G: nx.Graph) -> str:
    """判断边的距离属性键：优先 'length'（米），否则尝试 'length_m' 或其他。"""
    for _, _, data in G.edges(data=True):
        for k in ("length", "length_m", "distance", "weight"):
            if k in data:
                return k
        break
    return "length"


def build_distance_matrix(G: nx.Graph, node_ids: np.ndarray, wkey: str) -> np.ndarray:
    """
    基于路网最短路构建距离矩阵。
    G 已经过 largest_undirected_component 处理，理论上任意两点连通。
    """
    n = len(node_ids)
    dist = np.zeros((n, n), dtype=float)

    nodes_in_graph = set(G.nodes)
    missing = [nid for nid in node_ids if nid not in nodes_in_graph]
    if missing:
        raise ValueError(f"这些节点不在图中（可能映射或连通分量有问题）: {missing[:10]} ...")

    print(f"[Info] Building distance matrix for {n} nodes ...")

    for i, u in enumerate(node_ids):
        # 以 u 为源点，求到所有节点的最短路
        lengths = nx.single_source_dijkstra_path_length(G, u, weight=wkey)
        for j, v in enumerate(node_ids):
            if i == j:
                continue
            d = lengths.get(v, np.inf)
            dist[i, j] = float(d)

    # 检查是否还有 inf
    inf_mask = ~np.isfinite(dist)
    np.fill_diagonal(inf_mask, False)
    if inf_mask.any():
        n_inf = inf_mask.sum()
        print(f"[WARN] Distance matrix still has {n_inf} infinite entries，"
              f"说明这些候选点可能不在同一连通分量。")

    finite_ratio = np.isfinite(dist).mean()
    print("Distance matrix shape:", dist.shape,
          "| finite ratio (including diagonal):", finite_ratio)
    return dist


def build_tsp_graph(dist: np.ndarray, sub: pd.DataFrame) -> nx.Graph:
    """
    把距离矩阵转成 TSP 用的加权无向完全图；
    边属性包含 weight 和 length（两者相同，单位米）。
    """
    Gt = nx.Graph()
    n = dist.shape[0]

    # 节点属性
    for i in range(n):
        attrs = {"candidate_id": int(sub.loc[i, "cid"])}
        for k in ("lon", "lat", "x", "y", "osmid"):
            if k in sub.columns:
                val = sub.loc[i, k]
                try:
                    attrs[k] = float(val) if k in ("lon", "lat", "x", "y") else val
                except Exception:
                    attrs[k] = val
        Gt.add_node(i, **attrs)

    # 边权：强制完全图
    for i in range(n):
        for j in range(i + 1, n):
            w = dist[i, j]
            if not np.isfinite(w):
                # 理论上不该发生，兜底给一个极大值
                w = 1e12
            Gt.add_edge(i, j, weight=float(w), length=float(w))

    return Gt


# -----------------------
# 主流程
# -----------------------
if __name__ == "__main__":
    # size 表示合并前多少组解一起构建图
    size = 1

    # 1) 路网：加载 + 取最大连通分量（保证连通）
    G = pick_graph(GRAPHML_DENSIFIED, GRAPHML_RAW)
    G = largest_undirected_component(G)
    wkey = edge_weight_key(G)
    print(f"Loaded graph (LCC) with |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, weight_key='{wkey}'")

    # 2) 解编号（前 size 组解合并）
    indices = load_solution_indices(SOL_JSON, size)
    print(f"Combined solution size (first {size} groups): {len(indices)}")

    # 3) 候选点表（自动识别列名并规范）
    df_raw = load_candidates_df(CSV_PATTERN)
    df = normalize_columns(df_raw)

    # 4) 过滤本次用到的点，并映射到路网节点
    sub = filter_subset(df, indices)
    node_ids = map_points_to_nodes(G, sub)
    print(f"Mapped {len(node_ids)} points to graph nodes.")

    # 5) 路网最短路距离矩阵（理论上无 inf）
    dist = build_distance_matrix(G, node_ids, wkey=wkey)

    # 6) 构建 TSP 完全图
    G_tsp = build_tsp_graph(dist, sub)
    print(f"TSP graph built: |V|={G_tsp.number_of_nodes()}, |E|={G_tsp.number_of_edges()} (complete graph)")

    # 7) 保存为 GEXF
    os.makedirs("outputs/data", exist_ok=True)
    out_path = f"outputs/data/tsp_graph_{size}_solution.gexf"
    nx.write_gexf(G_tsp, out_path)
    print(f"TSP graph saved to: {out_path}")
