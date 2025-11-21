# -*- coding: utf-8 -*-
"""
可视化：在 GraphML 路网上标注 JSON 前 size 组点（来自 CSV 的坐标），并导出 PNG。
依赖：pandas, matplotlib, networkx, shapely
"""

import json
import os
from math import isfinite

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
import pandas as pd
from shapely import wkt as shapely_wkt

# ====== 修改为你的文件路径（已按你当前文件名填写）======
# size 表示：使用 JSON 里前 size 组解，合并这些点并可视化
size = 10
GRAPHML_PATH = "outputs/maps/marseille_1er_road_network.graphml"
CSV_PATH     = "outputs/data/marseille_1er_graph_nodes_projected.csv"
JSON_PATH    = "outputs/data/random_solutions_prepared.json"
# ===================================================


# ---------- 小工具函数 ----------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return x


def fix_graph_types(G: nx.Graph) -> nx.Graph:
    """
    统一 graph 属性的类型（x/y/lon/lat/length 等转成 float），
    并将字符串 geometry 尝试解析为 shapely.
    """
    for _, data in G.nodes(data=True):
        for k in ("x", "y", "lon", "lat"):
            if k in data:
                data[k] = _safe_float(data[k])

    # 根据图类型选择遍历方式
    if G.is_multigraph():
        iterator = G.edges(keys=True, data=True)
    else:
        iterator = ((u, v, None, d) for u, v, d in G.edges(data=True))

    for u, v, k, data in iterator:
        for wkey in ("length", "length_m", "distance", "weight"):
            if wkey in data:
                data[wkey] = _safe_float(data[wkey])
        if "geometry" in data and isinstance(data["geometry"], str):
            try:
                data["geometry"] = shapely_wkt.loads(data["geometry"])
            except Exception:
                pass
    return G


def load_first_n_solution_indices(json_path: str, size: int):
    """
    从 JSON 里读取前 size 组解，合并并去重，返回一个编号列表。

    JSON 结构期望是：
    [
        [1043, 1050, ...],   # 第 1 组
        [796, 1275, ...],    # 第 2 组
        ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        sols = json.load(f)

    if not sols:
        return []

    if size > len(sols):
        raise ValueError(f"size={size} 超过 JSON 中解的总数 {len(sols)}")

    combined = set()
    for group in sols[:size]:
        combined.update(group)

    # 排个序，便于复现和阅读
    return sorted(combined)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名，输出列：cid, lon, lat, x, y, osmid（按存在情况）"""
    low = {c.lower(): c for c in df.columns}
    id_col = None
    for k in ("id_index", "node_index", "candidate_id", "cid", "id"):
        if k in low:
            id_col = low[k]
            break
    if id_col is None:
        raise KeyError("CSV 未找到编号列（期望: id_index/node_index/candidate_id/cid/id）")

    lon_col = next((low.get(k) for k in ("lon", "longitude", "x") if k in low), None)
    lat_col = next((low.get(k) for k in ("lat", "latitude", "y") if k in low), None)
    x_col   = next((low.get(k) for k in ("x", "lon", "longitude") if k in low), None)
    y_col   = next((low.get(k) for k in ("y", "lat", "latitude") if k in low), None)
    osmid_col = next((low.get(k) for k in ("osmid", "osmid_node", "node", "node_id") if k in low), None)

    rename_map = {id_col: "cid"}
    if lon_col:   rename_map[lon_col] = "lon"
    if lat_col:   rename_map[lat_col] = "lat"
    if x_col:     rename_map[x_col] = "x"
    if y_col:     rename_map[y_col] = "y"
    if osmid_col: rename_map[osmid_col] = "osmid"

    out = df.rename(columns=rename_map).copy()
    return out


def filter_subset(df: pd.DataFrame, indices):
    sub = df[df["cid"].isin(indices)].copy()
    if sub.empty:
        raise ValueError("前 size 组解的编号在 CSV 中未匹配到任何行")
    sub = sub.sort_values(by="cid").reset_index(drop=True)
    return sub


def graph_is_lonlat(G: nx.Graph) -> bool:
    """简单判断节点坐标是否像经纬度范围"""
    xs = [d.get("x") for _, d in G.nodes(data=True) if "x" in d]
    ys = [d.get("y") for _, d in G.nodes(data=True) if "y" in d]
    xs = [x for x in xs if isinstance(x, (int, float)) and isfinite(x)]
    ys = [y for y in ys if isinstance(y, (int, float)) and isfinite(y)]
    if not xs or not ys:
        return True  # 缺值时默认当作经纬度
    return (min(xs) >= -180 and max(xs) <= 180) and (min(ys) >= -90 and max(ys) <= 90)


# ---------- 主函数 ----------
def main():
    # 1) 读 GraphML
    G = nx.read_graphml(GRAPHML_PATH)
    G = G.to_undirected()
    G = fix_graph_types(G)

    # 2) 读 JSON 的前 size 组 id_index，合并 & 去重
    indices = load_first_n_solution_indices(JSON_PATH, size=size)

    # 3) 读 CSV 并筛选对应点
    df_raw = pd.read_csv(CSV_PATH)
    df = normalize_columns(df_raw)
    sub = filter_subset(df, indices)

    # 4) 选择坐标列
    use_lonlat = graph_is_lonlat(G)
    if use_lonlat:
        if {"lon", "lat"}.issubset(sub.columns):
            px, py = sub["lon"].to_numpy(), sub["lat"].to_numpy()
        elif {"x", "y"}.issubset(sub.columns):
            px, py = sub["x"].to_numpy(), sub["y"].to_numpy()
        else:
            raise ValueError("CSV 未提供可用坐标列（lon/lat 或 x/y）")
    else:
        if {"x", "y"}.issubset(sub.columns):
            px, py = sub["x"].to_numpy(), sub["y"].to_numpy()
        elif {"lon", "lat"}.issubset(sub.columns):
            px, py = sub["lon"].to_numpy(), sub["lat"].to_numpy()
        else:
            raise ValueError("CSV 未提供可用坐标列（x/y 或 lon/lat）")

    # 5) 绘图
    fig, ax = plt.subplots(figsize=(12, 12))

    # 画边（统一黑色）
    if G.is_multigraph():
        edge_iter = G.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, None, d) for u, v, d in G.edges(data=True))

    for u, v, k, data in edge_iter:
        geom = data.get("geometry", None)
        if geom is not None and hasattr(geom, "coords"):
            xs, ys = zip(*geom.coords)
            ax.plot(xs, ys, linewidth=0.5, alpha=0.7, zorder=1, color="black")
        else:
            xu = G.nodes[u].get("x"); yu = G.nodes[u].get("y")
            xv = G.nodes[v].get("x"); yv = G.nodes[v].get("y")
            if None not in (xu, yu, xv, yv):
                ax.plot([xu, xv], [yu, yv], linewidth=0.5, alpha=0.7, zorder=1, color="black")

    # 画点（红色）
    sc = ax.scatter(px, py, s=90, c="red", edgecolors="black", linewidths=0.7, zorder=10)

    # 计算偏移量（按图幅尺寸的 1% 左右）
    if len(px) > 1:
        dx = (max(px) - min(px)) * 0.01
        dy = (max(py) - min(py)) * 0.01
    else:
        dx, dy = 0.001, 0.001

    # 6) 标注 cid，略作偏移 + 白色描边提高清晰度
    text_effect = [pe.withStroke(linewidth=2.5, foreground="white")]
    for i, row in sub.iterrows():
        x = px[i]
        y = py[i]
        cid = row["cid"]
        ax.text(
            x + dx, y + dy, str(int(cid)),
            fontsize=9, weight="bold", zorder=20,
            path_effects=text_effect
        )

    # 视窗范围留白（考虑标注）
    xpad = (max(px) - min(px)) * 0.08 if len(px) > 1 else 0.005
    ypad = (max(py) - min(py)) * 0.08 if len(py) > 1 else 0.005
    ax.set_xlim(min(px) - xpad, max(px) + xpad)
    ax.set_ylim(min(py) - ypad, max(py) + ypad)

    ax.set_title(f"First {size} solution groups (merged) on Marseille 1er road network")
    ax.set_xlabel("X (lon or meters)")
    ax.set_ylabel("Y (lat or meters)")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    # 7) 导出
    PNG_OUT = f"outputs/maps/TSP_Graph_first_{size}_groups.png"
    out_dir = os.path.dirname(PNG_OUT)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(PNG_OUT, dpi=240, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"✅ 图已保存到: {PNG_OUT}")


if __name__ == "__main__":
    main()
