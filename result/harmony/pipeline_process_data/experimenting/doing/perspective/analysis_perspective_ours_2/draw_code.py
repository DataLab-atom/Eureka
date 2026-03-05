import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    # 'font.size': 12,               
    'axes.labelsize': 18,          
    'axes.titlesize': 21,          
    'xtick.labelsize': 14,         
    'ytick.labelsize': 14,         
    'legend.fontsize': 18,         
    'font.family': 'Arial'
})
colors = ['#2c6f86', '#d0ead5', '#b5e2e5', '#57b35f', '#e6c7df', '#f79691']

# 示例数据
x1 = np.load('./experimental_result_data/hh_lisi_AC.npy')
y1 = np.load('./experimental_result_data/hh_lisi_AB.npy')

x2 = np.load('./experimental_result_data/ho_lisi_AC.npy')
y2 = np.load('./experimental_result_data/ho_lisi_AB.npy')

# 取第一个点
x_first, y_first = x1[0], y1[0]
# 取最后一个点
x_last1, y_last1 = x1[-1], y1[-1]
x_last2, y_last2 = x2[-1], y2[-1]



def draw_arrowed_line(ax, x, y, color):
    for i in range(3):
        # 这里画每条线段带箭头
        arrow = FancyArrowPatch((x[i], y[i]), (x[i+1], y[i+1]),
                                arrowstyle='->', color=color, mutation_scale=15,
                                linewidth=1.5, alpha=0.7)
        ax.add_patch(arrow)

fig, ax = plt.subplots(figsize=(7, 6))

ax.grid(True)
ax.set_facecolor('white')

ax.set_xlim(1.45, 1.90)
ax.set_ylim(1.20, 1.90)

xticks = ax.get_xticks()
yticks = ax.get_yticks()

# 构造隔一个标注一个的标签数组
xticklabels = [f'{tick:.2f}' if i % 2 != 0 else '' for i, tick in enumerate(xticks)]
yticklabels = [f'{tick:.2f}' if i % 2 != 0 else '' for i, tick in enumerate(yticks)]


# 应用到坐标轴
ax.set_xticks(xticks)
ax.set_xticklabels([1.45, '', '', 1.60, '', 1.70, '', '', '', 1.90])

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.set_ylim(1.2, 2.0)

ax.set_xlabel('Dataset A-B Average LISI', labelpad=16)
ax.set_ylabel('Dataset A-C Average LISI', labelpad=16)

ax.axvline(x=x_first, ymin=0, ymax=(y_first-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]), 
            color=colors[0], linestyle=':', linewidth=2, alpha=0.7)
ax.axhline(y=y_first, xmin=0, xmax=(x_first-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]), 
            color=colors[0], linestyle=':', linewidth=2, alpha=0.7)


ax.axvline(x=x_last1, ymin=0, ymax=(y_last1-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]), 
            color=colors[0], linestyle=':', linewidth=2, alpha=0.7)
ax.axhline(y=y_last1, xmin=0, xmax=(x_last1-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]), 
            color=colors[0], linestyle=':', linewidth=2, alpha=0.7)


ax.axvline(x=x_last2, ymin=0, ymax=(y_last2-ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]), 
            color=colors[3], linestyle=':', linewidth=2, alpha=0.7)
ax.axhline(y=y_last2, xmin=0, xmax=(x_last2-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]), 
            color=colors[3], linestyle=':', linewidth=2, alpha=0.7)

ax.grid(True, alpha=0.7, linewidth=1.5, color='#CCCCCC', linestyle='--')

# 标注
ax.text(x_first, ax.get_ylim()[0] - 0.02, f'{x_first:.3f}', color=colors[0], ha='center', va='top', fontsize=14, fontweight='bold', alpha=0.7)
ax.text(ax.get_xlim()[0] - 0.0075, y_first, f'{y_first:.3f}', color=colors[0], ha='right', va='center', fontsize=14, fontweight='bold', alpha=0.7)
ax.text(x_last1, ax.get_ylim()[0] - 0.02, f'{x_last1:.3f}', color=colors[0], ha='center', va='top', fontsize=14, fontweight='bold', alpha=0.7)

ax.text(ax.get_xlim()[0] - 0.0075, y_last1, f'{y_last1:.3f}', color=colors[0], ha='right', va='center', fontsize=14, fontweight='bold', alpha=0.7)

ax.text(x_last2, ax.get_ylim()[0] - 0.02, f'{x_last2:.3f}', color=colors[3], ha='center', va='top', fontsize=14, fontweight='bold', alpha=0.7)
ax.text(ax.get_xlim()[0] - 0.0075, y_last2, f'{y_last2:.3f}', color=colors[3], ha='right', va='center', fontsize=14, fontweight='bold', alpha=0.7)

# 绘制前4个点带箭头的线段
draw_arrowed_line(ax, x1, y1, colors[0])
draw_arrowed_line(ax, x2, y2, colors[3])
 
# 绘制后续点的普通直线
# 这里需要保证后续点数量>=2才画线
if len(x1) > 4:
    ax.plot(x1[3:], y1[3:], marker='o', linestyle='-.', color=colors[0], markersize=10)
 
if len(x2) > 4:
    ax.plot(x2[3:], y2[3:], marker='^', linestyle='-.', color=colors[3], markersize=10)
 
# 也绘制前4个点的散点，但去掉线条样式避免重复线
ax.scatter(x1[:4], y1[:4], marker='o', color=colors[0], s=80, alpha=0.7)
ax.scatter(x2[:4], y2[:4], marker='^', color=colors[3], s=80, alpha=0.7)

legend_elements = [
    Line2D([0], [0], color=colors[0], marker='o', linestyle='-', label='Harmony', markersize=10),
    Line2D([0], [0], color=colors[3], marker='^', linestyle='-', label='MoE-Harmony', markersize=10)
]

ax.legend(handles=legend_elements, loc='upper left')

ax.set_title('Average LISI(Iteration 15)', pad=15, fontweight='bold')

plt.savefig('./charts/plot.png', dpi=600, bbox_inches='tight')
















