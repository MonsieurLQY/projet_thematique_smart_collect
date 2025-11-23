# 项目说明 / Project Description

按顺序操作，足以复现这个项目。  
1 提取道路地图网格  
2 将道路网格中间均匀插入格点后，返回所有格点的坐标和序列号。  
3 生成随机格点Json文件。  
4 自定义尺寸大小，生成距离矩阵npy。  
5 自定义尺寸大小，将格点在地图上标出。

Follow the steps in order to fully reproduce this project.  
1 Extract the road map grid.  
2 After uniformly inserting grid points into the road grid, return the coordinates and serial numbers (IDs) of all grid points.  
3 Generate a JSON file of random grid points.  
4 Generate a GEXF graph with custom dimensions.  
5 Mark the grid points on the map with custom dimensions.

---
## 1_Extract_graph_map_Marseille_1st_Arrondismt

input:
    #Set the area of interest (1st arrondissement of Marseille)
    place_name = "1er arrondissement, Marseille, France"

output:
    marseille_1er_road_network(GraphML and png)

## 2_add_points_to_road

input :
    marseille_1er_road_network(GraphML in 1)

output :
    marseille_1er_densified_network(GraphML and png)  # road network with more inserted nodes 
    
    marseille_1er_graph_nodes_projected(csv)  # table about the information of all nodes 

## 3_generate_candidate_indices

input :
    modify the scale of candidates (the table in 2 shows the largest scale, which is the total number of nodes)

output :
    random_solutions(json)  # set of index of nodes 
    # 20 groupes, each group contains 10 nodes, you can modify it 

## 4_build_dixtances_matrix

input :
    marseille_1er_road_network(GraphML)  # road network in 1 
    marseille_1er_graph_nodes_projected(csv)  # table of nodes in 2 
    random_solutions_prepared(json)  # set of candidates of nodes in 3 (change the name of the file !!!)
    the scale you want (from 1 to 20)

output :
    distance_matrix_size(npy)
    distance_matrix_size_meta(json)

## 5_viz_TSP_Graph

input :
    marseille_1er_road_network(GraphML)  # road network in 1 
    marseille_1er_graph_nodes_projected(csv)  # table of nodes in 2 
    random_solutions_prepared(json)  # set of candidates of nodes in 3 (change the name of the file !!!)
    the scale you want (from 1 to 20)

output :
    TSP_Graph_first_{size}_groups (png)