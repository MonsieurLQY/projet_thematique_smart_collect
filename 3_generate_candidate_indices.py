import json
import random
import os


# ！！！ attention: the name of the output file should be consistent with that in 4_build_distances_matrix.py  !!!
def generate_random_solutions(output_path="solutions.json", num_groups=20, group_size=10, value_range=(0, 1292)):
    """
    生成随机解文件，每组包含若干随机整数。
    :param output_path: 输出 JSON 文件路径
    :param num_groups: 组数（默认 20）
    :param group_size: 每组整数数量（默认 10）
    :param value_range: 随机整数范围 (min, max)
    """
    min_val, max_val = value_range
    solutions = []

    for _ in range(num_groups):
        group = random.sample(range(min_val, max_val + 1), group_size)
        solutions.append(group)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(solutions, f, indent=4, ensure_ascii=False)

    print(f"✅ 成功生成随机解文件：{output_path}")
    print(f"包含 {num_groups} 组，每组 {group_size} 个整数。")

if __name__ == "__main__":
    # modify the name of output file as needed
    generate_random_solutions("outputs/data/random_solutions.json")
