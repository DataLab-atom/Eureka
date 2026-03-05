# tsp100 - 只绘制GA部分
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon

plt.rcParams['font.family'] = 'Arial'
colors = ['#88c4d7', '#d0ead5', '#b5e2e5', '#9793c6', '#e6c7df', '#f79691']

# ✅ 提取TSP100的数据
data = {
    'Type': ['GA', 'GA+EOH', 'GA+ReEvo', 'GA+U2E(ours)', 
             'ACO', 'ACO+EOH', 'ACO+ReEvo', 'ACO+U2E(ours)',
             'KGLS', 'KGLS+EOH', 'KGLS+ReEvo', 'KGLS+U2E(ours)'],
    'Obj': [40.8, 40.5, 40.6, 36.6, 8.5, 8.5, 8.4, 8.4, 9.3, 9.2, 9.3, 8.5],
    'time': [2.3, 2.0, 2.1, 1.3, 17.9, 17.4, 12.2, 13.7, 26.8, 28.8, 20.9, 28.0]
}

df = pd.DataFrame(data)

# 🔧 按方法类型分配颜色
method_colors = {
    'GA': colors[5],
    'GA+EOH': colors[1],
    'GA+ReEvo': colors[2],
    'GA+U2E(ours)': colors[3],
    # 'ACO': colors[5],
    # 'ACO+EOH': colors[1],
    # 'ACO+ReEvo': colors[2],
    # 'ACO+U2E(ours)': colors[3],
    # 'KGLS': colors[5],
    # 'KGLS+EOH': colors[1],
    # 'KGLS+ReEvo': colors[2],
    # 'KGLS+U2E(ours)': colors[3]
}

# 创建单幅散点图
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# 只筛选GA组的数据
group_methods = ['GA', 'GA+EOH', 'GA+ReEvo', 'GA+U2E(ours)']
group_data = df[df['Type'].isin(group_methods)]

# 设置坐标轴范围
ax.set_xlim(1.2, 2.4)
ax.set_ylim(0, 52)

# 获取当前设置后的坐标轴范围
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
origin = (x_min, y_min)  # 实际显示的原点

# 1. 找最小 Obj 的点（优先 U2E）
min_obj_rows = group_data[group_data['Obj'] == group_data['Obj'].min()]
min_point = min_obj_rows.loc[min_obj_rows['Type'].str.contains('U2E').idxmax()]  # 优先选含 U2E 的
min_x, min_y = min_point['time'], min_point['Obj']

# 2. 找最大 Obj 的点（优先 time 大的）
max_obj_rows = group_data[group_data['Obj'] == group_data['Obj'].max()]
max_point = max_obj_rows.loc[max_obj_rows['time'].idxmax()]  # 取 time 最大的
max_x, max_y = max_point['time'], max_point['Obj']

# 3. 构建三角形顶点
triangle_points = [(1.34, 52), (2.6, 52), origin]

# 4. 绘制三角形并填充阴影
polygon = Polygon(triangle_points, closed=True, fill=True, alpha=0.4, edgecolor=colors[4], facecolor=colors[4])
ax.add_patch(polygon)

# 5. 绘制从原点出发、穿过每个点、延伸到坐标轴边界的射线
for idx, row in group_data.iterrows():
    x, y = row['time'], row['Obj']
    method_type = row['Type']

    # 射线方向向量
    dx = x - x_min
    dy = y - y_min

    if dx == 0 and dy == 0:
        continue  # 避免除零错误

    t = 1.5  # 固定延伸系数

    # 计算终点坐标
    end_x = x_min + t * dx
    end_y = y_min + t * dy

    # 绘制从原点到边界点的线段
    ax.plot([x_min, end_x], [y_min, end_y],
            color=method_colors[method_type], linestyle='-', alpha=0.8, linewidth=3)

# 6. 绘制散点图（按方法类型颜色）
for idx, row in group_data.iterrows():
    x, y = row['time'], row['Obj']
    method_type = row['Type']
    ax.scatter(x, y, c=method_colors[method_type], s=350, alpha=0.8, edgecolors='black', linewidth=0.5)

# 7. 添加方法标签
# for idx, row in group_data.iterrows():
#     x, y = row['time'], row['Obj']
#     method_type = row['Type']
    
#     # 手动调整每个标签的位置
#     if method_type == 'GA':
#         xytext = (5, 5)
#         ha, va = 'left', 'bottom'
#         end_method_type = 'GA'
#     elif method_type == 'GA+EOH':
#         xytext = (-40, 13)
#         ha, va = 'left', 'top'
#         end_method_type = 'GA+EOH'
#     elif method_type == 'GA+ReEvo':
#         xytext = (-5, 5)
#         ha, va = 'right', 'bottom'
#         end_method_type = 'GA+ReEvo'
#     elif method_type == 'GA+U2E(ours)':
#         xytext = (5, 5)
#         ha, va = 'left', 'bottom'
#         end_method_type = 'GA+U2E(ours)'
    
    # ax.annotate(end_method_type, (x, y), 
    #             xytext=xytext, textcoords='offset points',
    #             fontsize=12, ha=ha, va=va, 
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 设置坐标轴标签
ax.set_xlabel('Time', fontsize=18)
ax.set_ylabel('Objective Value', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=12)

# 添加网格
ax.grid(True, alpha=0.7, ls='--', linewidth=1.5, color='#CCCCCC')

# 添加图例
from matplotlib.patches import Patch

legend_colors = {
    'Original': colors[5],
    'EOH': colors[1],
    'ReEvo': colors[2],
    'Ours': colors[3],
    'Best-worst difference': colors[4]
}

legend_elements = [
    Patch(facecolor=legend_colors['Original'], edgecolor='black', label='Original'),
    Patch(facecolor=legend_colors['EOH'], edgecolor='black', label='EOH'),
    Patch(facecolor=legend_colors['ReEvo'], edgecolor='black', label='ReEvo'),
    Patch(facecolor=legend_colors['Ours'], edgecolor='black', label='U2E'),
    Patch(facecolor=legend_colors['Best-worst difference'], edgecolor='black', label='Best-worst difference'),
]

fig.legend(handles=legend_elements, 
           loc='lower center', 
           ncol=5, 
           fontsize=13,
           bbox_to_anchor=(0.53, -0.06))  # 调整 vertical 位置

# 调整布局
plt.tight_layout()

# 保存图表（可选）
for mod in ['pdf', 'svg', 'png']:
    plt.savefig(f"tsp_100/GA_only_plot.{mod}", dpi=600, bbox_inches='tight')

plt.show()