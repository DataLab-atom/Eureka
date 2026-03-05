# tsp50
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon

plt.rcParams['font.family'] = 'Arial'
colors = ['#88c4d7', '#d0ead5', '#b5e2e5', '#9793c6', '#e6c7df', '#f79691']

# 提取TSP50的数据
data = {
    'Type': ['GA', 'GA+EOH', 'GA+ReEvo', 'GA+U2E(ours)', 
             'ACO', 'ACO+EOH', 'ACO+ReEvo', 'ACO+U2E(ours)',
             'KGLS', 'KGLS+EOH', 'KGLS+ReEvo', 'KGLS+U2E(ours)'],
    'Obj': [18.2, 20, 17.9, 16.3, 6.0, 5.9, 5.9, 5.8, 6.7, 6.8, 6.8, 5.9], # [18.2, 17.8, 17.9, 16.3, 5.9, 5.9, 5.9, 5.8, 6.7, 6.8, 6.8, 5.9]
    'time': [1.3, 0.84, 0.8, 0.6, 7.8, 9.1, 7.6, 6.4, 10.3, 14.0, 14.9, 9.0]  # [1.3, 0.8, 0.8, 0.6, 7.6, 9.1, 7.6, 6.4, 10.3, 14.0, 14.9, 9.0]
}

df = pd.DataFrame(data)

# 定义解法组
groups = {
    'GA': ['GA', 'GA+EOH', 'GA+ReEvo', 'GA+U2E(ours)'],
    'ACO': ['ACO', 'ACO+EOH', 'ACO+ReEvo', 'ACO+U2E(ours)'],
    'KGLS': ['KGLS', 'KGLS+EOH', 'KGLS+ReEvo', 'KGLS+U2E(ours)']
}

# 🔧 按方法类型分配颜色（相同方法类型用相同颜色）
method_colors = {
    'GA': colors[5],
    'GA+EOH': colors[1],
    'GA+ReEvo': colors[2],
    'GA+U2E(ours)': colors[3],
    'ACO': colors[5],
    'ACO+EOH': colors[1],
    'ACO+ReEvo': colors[2],
    'ACO+U2E(ours)': colors[3],
    'KGLS': colors[5],
    'KGLS+EOH': colors[1],
    'KGLS+ReEvo': colors[2],
    'KGLS+U2E(ours)': colors[3]
}

# 创建三幅散点图
fig, axes = plt.subplots(1, 3, figsize=(14, 7))

kkk=[[(0.679, 22), (1.5, 22)], 
     [(6.55, 6.1), (9.75, 6.1)],
     [(9.4, 8), (16.1, 8)]]

for i, (group_name, group_methods) in enumerate(groups.items()):
    ax = axes[i]
    
    # 筛选当前组的数据
    group_data = df[df['Type'].isin(group_methods)]
    
    # 先设置坐标轴范围
    if group_name == 'GA':
        ax.set_xlim(0.4, 1.4)
        ax.set_ylim(0, 19)
    elif group_name == 'ACO':
        ax.set_xlim(6, 9.5)
        ax.set_ylim(5, 5.95)
    elif group_name == 'KGLS':
        ax.set_xlim(8, 15.5)
        ax.set_ylim(0, 7)
    
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
    triangle_points = [kkk[i][0], kkk[i][1], origin]
    # triangle_points = [(min_x, min_y), (max_x, max_y), origin]

    # 4. 绘制三角形并填充阴影（使用组的颜色）
    polygon = Polygon(triangle_points, closed=True, fill=True, alpha=0.4, edgecolor=colors[4], facecolor=colors[4])  # edgecolor=method_colors[group_methods[0]], facecolor=method_colors[group_methods[0]]
    ax.add_patch(polygon)
    

    # 5. 绘制从原点出发、穿过每个点、延伸到坐标轴边界的射线（触顶）
    for idx, row in group_data.iterrows():
        x, y = row['time'], row['Obj']
        method_type = row['Type']

        # 射线方向向量
        dx = x - x_min
        dy = y - y_min

        if dx == 0 and dy == 0:
            continue  # 避免除零错误

        # 计算射线与坐标轴边界的交点
        # 我们要找的是射线第一次碰到 x=x_max 或 y=y_max 的点
        t_x = (x_max - x_min) / dx if dx != 0 else float('inf')
        t_y = (y_max - y_min) / dy if dy != 0 else float('inf')

        # 取最小正 t 值（即最先碰到的边界）
        # t = min(t for t in [t_x, t_y] if t > 0)
        t=1.5

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
    
    # 7. 添加方法标签（手动调偏移）
    for idx, row in group_data.iterrows():
        x, y = row['time'], row['Obj']
        method_type = row['Type']
        
        # 手动调整每个标签的位置
        if group_name == 'GA':

            ax.set_ylim(0, 22)
            # ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])

            # if method_type == 'GA':
            #     xytext = (5, 5)
            #     ha, va = 'left', 'bottom'
            #     end_method_type=''  # 'Original', 'EOH', 'ReEvo', 'Ours'
            # elif method_type == 'GA+EOH':
            #     xytext = (-40, 13)  # 向右下偏移
            #     ha, va = 'left', 'top'
            #     end_method_type='EOH'
            # elif method_type == 'GA+ReEvo':
            #     xytext = (-5, 5)  # 向左上偏移
            #     ha, va = 'right', 'bottom'
            #     end_method_type=''
            # elif method_type == 'GA+U2E(ours)':
            #     xytext = (5, 5)
            #     ha, va = 'left', 'bottom'
            #     end_method_type=''
                
        elif group_name == 'ACO':

            ax.set_ylim(5, 6.1)
            
            # ax.set_xticks([6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5])

            # if method_type == 'ACO':
            #     xytext = (-57, 15)
            #     ha, va = 'left', 'top'
            #     end_method_type='Original'
            # elif method_type == 'ACO+EOH':
            #     xytext = (5, -5)  # 向右下偏移
            #     ha, va = 'left', 'top'
            #     end_method_type=''
            # elif method_type == 'ACO+ReEvo':
            #     xytext = (-15, -3)  # 向左上偏移
            #     ha, va = 'right', 'bottom'
            #     end_method_type=''
            # elif method_type == 'ACO+U2E(ours)':
            #     xytext = (5, 5)
            #     ha, va = 'left', 'bottom'
            #     end_method_type=''
                
        elif group_name == 'KGLS':
        
            ax.set_ylim(0, 8)
            # ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])

            # if method_type == 'KGLS':
            #     xytext = (5, 5)
            #     ha, va = 'left', 'bottom'
            #     end_method_type=''
            # elif method_type == 'KGLS+EOH':
            #     xytext = (5, -5)  # 向右下偏移
            #     ha, va = 'left', 'top'
            #     end_method_type=''
            # elif method_type == 'KGLS+ReEvo':
            #     xytext = (-5, 5)  # 向左上偏移
            #     ha, va = 'right', 'bottom'
            #     end_method_type=''
            # elif method_type == 'KGLS+U2E(ours)':
            #     xytext = (5, 5)
            #     ha, va = 'left', 'bottom'
            #     end_method_type=''
        
    
        # ax.annotate(end_method_type, (x, y), 
        #            xytext=xytext, textcoords='offset points',
        #            fontsize=12, ha=ha, va=va, 
        #            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # 设置坐标轴标签
    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel(f'Objective Value({group_name})', fontsize=18)
    # ax.set_title(f'{group_name}', fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 添加网格
    ax.grid(True, alpha=0.7, ls='--', linewidth=1.5, color='#CCCCCC')

# 调整布局
plt.tight_layout()
# 在 plt.tight_layout() 之后、plt.show() 之前添加：
# fig.suptitle('TSP50', fontsize=22, fontweight='bold', y=1)


from matplotlib.patches import Patch

# 定义统一的颜色映射（与你的 method_colors 逻辑一致）
legend_colors = {
    'Original': colors[5],   # 对应基础方法（GA, ACO, KGLS）
    'EOH':      colors[1],   # +EOH
    'ReEvo':    colors[2],   # +ReEvo
    'Ours':     colors[3]  ,  # +U2E(ours)
    'Best-worst difference': colors[4]
}

# 创建图例句柄
legend_elements = [
    Patch(facecolor=legend_colors['Original'], edgecolor='black', label='Original'),
    Patch(facecolor=legend_colors['EOH'],      edgecolor='black', label='EOH'),
    Patch(facecolor=legend_colors['ReEvo'],    edgecolor='black', label='ReEvo'),
    Patch(facecolor=legend_colors['Ours'],     edgecolor='black', label='U2E'),
    Patch(facecolor=legend_colors['Best-worst difference'],     edgecolor='black', label='Best-worst difference'),
]

# 添加统一图例（放在底部下方）
fig.legend(handles=legend_elements, 
           loc='lower center', 
           ncol=5, 
           fontsize=18,
           bbox_to_anchor=(0.5, 0))  # 调整 vertical 位置

# 重新调整布局，避免图例被裁剪
plt.tight_layout(rect=[0, 0.08, 1, 1])  # 给底部留出空间（0.05 是底部 margin） #rect=[0, 0.04, 1, 1]

for mod in ['pdf', 'svg', 'png']:
    plt.savefig(f"tsp_50/plot.{mod}", dpi=600)

plt.show()

# 保存图表
# plt.savefig('tsp50_scatter_plots_with_triangle.png', dpi=300, bbox_inches='tight')
