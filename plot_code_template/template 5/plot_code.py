import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# from pyparsing import alphas

# === Set academic paper style parameters ===
# plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Times New Roman',  # Using Times New Roman font
    'font.size': 9,          # Base font size
    # 'axes.edgecolor': '#333333',
    'axes.titlesize': 12,    # Axis title font size
    'axes.labelsize': 10,     # Axis label font size
    'xtick.labelsize': 7.5,    # X axis tick font size
    'ytick.labelsize': 7.5,    # Y axis tick font size
    'legend.fontsize': 9,    # Legend font size
    'figure.dpi': 700,       # High resolution
    'figure.figsize': (6, 6),  # Standard academic figure size
    'lines.linewidth': 3.5,  # Line width
    'axes.linewidth': 1,   # Axis line width
    'grid.linewidth': 3.5,       # ✅ 加粗网格线：从 0.4 → 0.7
    'grid.alpha': 0.8,           # ✅ 增强网格透明度，使其更清晰
    'grid.color': "#cccccc",     # ✅ 可选：设置网格颜色为浅灰
    'grid.linestyle': '--',       # ✅ 可选：实线网格（默认是虚线 '-.' 或 ':'）
    'xtick.direction': 'out',   # ✅ X轴刻度线朝外
    'ytick.direction': 'out',   # ✅ Y轴刻度线朝外
     'xtick.major.size': 6,      # ✅ 主刻度线长度（像素）
    'ytick.major.size': 6,
    'xtick.major.width': 3,   # X轴主刻度线宽度
    'ytick.major.width': 3,   # Y轴主刻度线宽度
})

# Define color scheme
#colors = ['#88c4d7', '#9793c6', '#f79691', '#66c1a4']  # Baseline: blueish, Our method: purple, Marker: reddish

# Define line styles
line_styles = ['-', '-.']  # Baseline solid, Ours dashed

# Data directory and output directory
import os
root_path=os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_path, 'draw_data')
out_dir= os.path.join(root_path, 'charts')
# out_dir = f'/data/zz/rag_paper/experiment_auto_write/result/pgrr/angle/analysis_angle_001/charts'

# Make sure output directory exists
os.makedirs(out_dir, exist_ok=True)

# File pairs of baseline and proposed data
file_pairs = [
    ('draw_data_base.npy', 'draw_data_our.npy')
    ('draw_data_000.npy', 'draw_data_001.npy'),
    ('draw_data_002.npy', 'draw_data_003.npy'),
    ('draw_data_004.npy', 'draw_data_005.npy'),
]

colors = plt.cm.viridis(np.linspace(0, 1, 4))

# We will generate separate charts for each pair

for idx, (base_file, ours_file) in enumerate(file_pairs):
    baseline = np.load(os.path.join(data_dir, base_file))/100
    ours = np.load(os.path.join(data_dir, ours_file))/100

    # X axis: iterations, assuming values are per p iterations
    iterations = np.arange(0, len(baseline)*5000, 5000)

    fig, ax = plt.subplots()

    # Plot baseline
    ax.plot(iterations, baseline, color=colors[0], linestyle=line_styles[0], label='PGRR', linewidth=6.5)

    # Plot ours
    ax.plot(iterations, ours, color=colors[1], linestyle=line_styles[1], label='Ours', linewidth=6.5)


    diff = baseline - ours


    # Plot difference curve: baseline - ours
    ax2 = ax.twinx()
    ax2.grid(False)  # 关闭右侧轴网格

    # ax2.set_ylabel('Error Difference (PGRR - Ours)', fontname='Times New Roman', fontsize=8, color=colors[2])
    # ax2.set_yticks([0, 0.016, 0.055])  # 手动设置刻度位置
    ax2.set_ylim(0, 0.7e-5)
    ax2.set_yticks([]) 
    # ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))


    # Plot difference curve on right axis, with low zorder to stay underneath
    
    ax2.plot(iterations, diff, color=colors[3], linestyle=':', linewidth=6.5, label='Error Difference (PGRR - Ours)')
    # 在差值曲线与 X 轴之间填充阴影（仅当 diff >= 0）
    ax2.fill_between(iterations, 0, diff, where=(diff >= 0), color=colors[3], alpha=0.4, label='Improvement Region (PGRR > Ours)')

        # 找到差值最大的索引
    max_diff_idx = np.argmax(diff)
    max_diff_iter = iterations[max_diff_idx]

    # 在主图 ax 上画一条竖直线（贯穿整个 y 轴范围）
    ax.axvline(x=max_diff_iter, color=colors[2], linestyle='--', linewidth=3, alpha=0.5, label=f'Max Improvement({max_diff_iter:,} iter)')

    # Axis labels
    ax.set_xlabel('Iterations', fontname='Times New Roman', fontsize=23, fontweight='bold')
    ax.set_ylabel('Error', fontname='Times New Roman', fontsize=23, fontweight='bold')

    # X axis settings
    ax.xaxis.set_major_locator(MaxNLocator(6))
    # Y axis settings (auto)
    ax.yaxis.set_major_locator(MaxNLocator(6))

    # Title with context
    # ax.set_title('Convergence Curves: PGRR vs Ours', fontname='Times New Roman', fontsize=19, pad=10, fontweight='bold')

    #Legend with border
    # ax.legend(frameon=True, loc='upper right', framealpha=0.7)
    # 获取两个轴的句柄和标签
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # 合并
    lines = lines1 + lines2
    labels = labels1 + labels2

    # 在主轴 ax 上绘制统一图例
    ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.415, -0.17), ncol=2, fontsize=17, prop={'weight': 'bold'})  
    # ax.legend(loc='upper center', bbox_to_anchor=(0.42, -0.17), ncol=3, fontsize=15, prop={'weight': 'bold'})

    # 使用线性坐标，清晰显示 0 线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.6, label='Zero Error Difference')

    ax.tick_params(labelsize=17)

    ax.set_yscale('log')

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin * 0.5, ymax)

   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.spines['left'].set_linewidth(2.5)
    ax2.spines['bottom'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)

    ax.grid(True)

    plt.tight_layout()

    # Save figure
    # save_path = os.path.join(out_dir, f'chart_{idx:03d}.png')
    plt.savefig(os.path.join(out_dir, f'chart_001.png'), dpi=700, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f'chart_001.svg'), dpi=700, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f'chart_001.pdf'), dpi=700, bbox_inches='tight')
    plt.close(fig)

# If no data files present, generate a placeholder plot to indicate no data
# But per instructions, if files not found, skip generating plot

