import osmnx as ox
import networkx as nx
from shapely.geometry import LineString, Point
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 定义项目路径（假设当前脚本与前面的脚本在同一项目目录）
# Define the project path (assuming this script is in the same project folder)
project_dir = Path(__file__).resolve().parent
graph_dir = project_dir / "outputs" / "maps"
data_dir = project_dir / "outputs" / "data"

# 确保输出文件夹存在
# Ensure output folders exist
graph_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# 加载先前生成的 GraphML 文件
# Load the previously saved GraphML file
graph_path = graph_dir / "marseille_1er_road_network.graphml"
G = ox.load_graphml(graph_path)

# 投影为米制坐标系（方便测距和插值）
# Project the graph to a metric CRS for distance calculation
G_proj = ox.project_graph(G, to_crs="EPSG:32631")
edges_proj = ox.graph_to_gdfs(G_proj, nodes=False, edges=True)

# 设置插值参数（每 25 米插一个点）
# Set interpolation interval (insert a point every 25 meters)
target_interval = 25  # meters
G_aug = G_proj.copy()
node_id_counter = max(G_aug.nodes) + 1

# 对每条边进行插值
# Densify each edge by adding interpolated nodes
for (u, v, k), row in edges_proj.iterrows():
    geometry: LineString = row.geometry
    length = geometry.length

    # 忽略短边 / Skip short edges
    if length <= target_interval:
        continue

    # 计算分段数 / Calculate the number of segments
    num_segments = max(1, round(length / target_interval))
    segment_length = length / num_segments

    # 生成插值点（包含首尾） / Generate equally spaced interpolation points
    points = [geometry.interpolate(i * segment_length) for i in range(num_segments + 1)]

    # 构建节点序列 / Build a sequence of nodes along the edge
    path_nodes = [u]
    for pt in points[1:-1]:
        new_node_id = node_id_counter
        node_id_counter += 1
        G_aug.add_node(new_node_id, x=pt.x, y=pt.y)
        path_nodes.append(new_node_id)
    path_nodes.append(v)

    # 逐段添加新边 / Add new subdivided edges
    for i in range(len(path_nodes) - 1):
        n1 = path_nodes[i]
        n2 = path_nodes[i + 1]

        if n1 in G_aug.nodes and n2 in G_aug.nodes:
            p1 = Point(G_aug.nodes[n1]["x"], G_aug.nodes[n1]["y"])
            p2 = Point(G_aug.nodes[n2]["x"], G_aug.nodes[n2]["y"])
            seg_len = p1.distance(p2)
            geom = LineString([p1, p2])
            G_aug.add_edge(n1, n2, length=seg_len, geometry=geom)

    # 删除原始长边 / Remove the original long edge
    if G_aug.has_edge(u, v, k):
        G_aug.remove_edge(u, v, k)

# 可视化加密后的图形（恢复原样式）
# Visualize the densified graph (using default OSMnx style)
fig, ax = ox.plot_graph(G_aug, node_size=5, show=False)

# 保存图片到 outputs/maps/
# Save the plotted figure to outputs/maps/
plot_path = graph_dir / "marseille_1er_densified_network.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")

# 显示图像在屏幕上
# Show the graph on screen
plt.show()

# 保存细分后的网络文件
# Save the densified graph as GraphML
densified_graph_path = graph_dir / "marseille_1er_densified_network.graphml"
ox.save_graphml(G_aug, filepath=densified_graph_path)

# —— 导出节点坐标到 CSV，包含投影坐标（米）、经纬度（度）、以及编号 ——
# —— Export node coordinates to CSV with projected meters, lon/lat, and sequential index ——

node_records = []
for node_id, data in G_aug.nodes(data=True):
    x_m = float(data["x"])
    y_m = float(data["y"])
    node_records.append({"osm_id": node_id, "x_meters": x_m, "y_meters": y_m})

# 转为 GeoDataFrame，声明当前投影为 EPSG:32631（米）
gdf_nodes = gpd.GeoDataFrame(
    node_records,
    geometry=[Point(rec["x_meters"], rec["y_meters"]) for rec in node_records],
    crs="EPSG:32631",
)

# 投影到 WGS84（EPSG:4326），获取经纬度
gdf_ll = gdf_nodes.to_crs("EPSG:4326")
gdf_nodes["lon"] = gdf_ll.geometry.x
gdf_nodes["lat"] = gdf_ll.geometry.y

# 添加连续编号列，从 0 开始
# Add sequential index column starting from 0
gdf_nodes = gdf_nodes.reset_index(drop=True)
gdf_nodes["node_index"] = gdf_nodes.index

# 调整列顺序：node_index 在最前面
# Reorder columns: node_index first
gdf_nodes = gdf_nodes[["node_index", "osm_id", "x_meters", "y_meters", "lon", "lat"]]

# 保存为 CSV 文件（输出到 outputs/data/）
csv_path = data_dir / "marseille_1er_graph_nodes_projected.csv"
gdf_nodes.to_csv(csv_path, index=False)

print(f"Densified graph saved at: {densified_graph_path}")
print(f"Densified network plot saved at: {plot_path}")
print(f"Projected node coordinates (meters) + lon/lat exported to: {csv_path}")
print(f"节点总数 (Total nodes): {len(G_aug.nodes)}")
