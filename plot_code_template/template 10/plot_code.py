import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

plt.rcParams['font.family'] = 'Arial'
colors = ['#88c4d7', '#d0ead5', '#b5e2e5', '#9793c6', '#e6c7df', '#f79691']

def draw_tsp_solution_with_arrows(node_positions, solution, title="TSP Solution"):
    """
    绘制带箭头的TSP路径图
    
    Parameters:
    node_positions: 节点坐标数组 (n, 2)
    solution: 路径序列 (list or array)
    title: 图表标题
    """
    # 创建路径坐标
    path_x = []
    path_y = []
    
    # 按照solution的顺序获取坐标
    for node_idx in solution:
        path_x.append(node_positions[int(node_idx), 0])
        path_y.append(node_positions[int(node_idx), 1])
    
    # 添加回到起点的连线，形成完整回路
    path_x.append(node_positions[int(solution[0]), 0])
    path_y.append(node_positions[int(solution[0]), 1])
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制所有节点
    ax.scatter(node_positions[:, 0], node_positions[:, 1], c=colors[3], s=120, zorder=5, label='Nodes')
    
    # 标记起始节点
    ax.scatter(node_positions[int(solution[0]), 0], node_positions[int(solution[0]), 1], 
               c=colors[5], s=120, marker='s', zorder=6, label='Start')
    
    # 在每段路径上画箭头
    for i in range(len(path_x) - 1):
        x_start = path_x[i]
        y_start = path_y[i]
        x_end = path_x[i + 1]
        y_end = path_y[i + 1]
        
        # 计算向量
        dx = x_end - x_start
        dy = y_end - y_start
        
        # 画箭头
        ax.quiver(x_start, y_start, dx, dy, 
                  angles='xy', scale_units='xy', scale=1,
                  color=colors[0], alpha=0.7, width=0.006, headwidth=6, headlength=6)
    
    ax.set_title(title, fontsize=20, pad=15, x=0.47, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=15, labelpad=10)
    ax.set_ylabel('Y Coordinate', fontsize=15, labelpad=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)

    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.5, lw=1.5, ls='--')
    ax.set_aspect('equal')  # 保持比例一致

    import os
    os.makedirs(f'tsp_20/{title}', exist_ok=True)

    plt.tight_layout()

    for mod in ['pdf', 'svg', 'png']:
        plt.savefig(f"tsp_20/{title}/{title}.{mod}", dpi=600, bbox_inches='tight')

    plt.show()

def calculate_tour_distance(node_positions, solution):
    """
    计算路径总距离
    """
    dist_mat = distance_matrix(node_positions, node_positions)
    total_distance = 0
    n = len(solution)
    
    for i in range(n):
        current_node = int(solution[i])
        next_node = int(solution[(i + 1) % n])
        total_distance += dist_mat[current_node, next_node]
    
    return total_distance

# 加载数据并绘图
def visualize_tsp_solutions(problem_size=20, mode='val', instance_idx=0, solutions_path=''):

    # 加载数据集
    dataset_path = "./val20_dataset.npy"
    node_positions_all = np.load(dataset_path)
    
    # 加载解决方案
    # solutions_path = "./reevo_val20_solutions.npy"
    solutions = np.load(solutions_path, allow_pickle=True)
    
    # 选择特定实例
    node_positions = node_positions_all[instance_idx]
    solution = solutions[instance_idx]
    
    # 计算路径长度
    distance = calculate_tour_distance(node_positions, solution)
        
    # 绘制路径图（带箭头）
    # title = f'TSP Solution - Size: {problem_size}, Mode: {mode}, Instance: {instance_idx}, Distance: {distance:.4f}'
    if 'u2e' in  solutions_path:
        title='U2E'
    elif 'baseline' in  solutions_path:
        title='Original'
    elif 'eoh' in  solutions_path:
        title='EoH'
    elif 'reevo' in  solutions_path:
        title='ReEvo'
    draw_tsp_solution_with_arrows(node_positions, solution, title)


dataset_list=[
 'u2e_val20_solutions.npy',
 'val20_baseline_solutions.npy',
 'eoh_val20_solutions.npy',
 'reevo_val20_solutions.npy'
 ]
for i in dataset_list:
    visualize_tsp_solutions(problem_size=20, mode='val', instance_idx=0, solutions_path=i)