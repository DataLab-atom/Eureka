import numpy as np

data_d = np.load('fig_d.npz')
list(data_d.keys())

import re
new_data_d = {}

for key,data in data_d.items():
    new_data_d[key] = [data.mean(),data.std()]
    if np.isnan(new_data_d[key][0]):
        new_data_d[key][0] = 0
    if np.isnan(new_data_d[key][1]):
        new_data_d[key][1] = 0
        
new_data_d_2d = {}
for key in new_data_d.keys():
    if 'reevo' in key:
        continue
    new_data_d_2d[re.sub('_f\d_l2','',key)] = [[],[]]


for key,data in new_data_d.items():
    if 'reevo' in key:
        continue
    new_data_d_2d[re.sub('_f\d_l2','',key)][0].append(data[0])
    new_data_d_2d[re.sub('_f\d_l2','',key)][1].append(data[1])
new_data_d_2d
new_data_d_2d_T = {f'$f_{i}$':[[],[]] for i in range(2,6)}
for key,data in new_data_d_2d.items():
    print(key)
    if key == 'reevo':
        continue
    for i in range(len(data[0])):
        new_data_d_2d_T[f'$f_{i+2}$'][0].append(data[0][i])
        new_data_d_2d_T[f'$f_{i+2}$'][1].append(data[1][i])
# new_data_d_2d_T


data={}
for idx in range(4):
    data[f"f{idx+1}"]={}
    for method in new_data_d_2d:
        if method == 'baseline':
            data[f"f{idx+1}"]['Baseline']=[new_data_d_2d[method][0][idx], new_data_d_2d[method][1][idx]]

        elif method == 'funsearch':
            data[f"f{idx+1}"]['FunSearch']=[new_data_d_2d[method][0][idx], new_data_d_2d[method][1][idx]]

        elif method == 'eoh':
            data[f"f{idx+1}"]['EoH']=[new_data_d_2d[method][0][idx], new_data_d_2d[method][1][idx]]

        elif method == 'alphaevolve':
            data[f"f{idx+1}"]['AlphaEvolve']=[new_data_d_2d[method][0][idx], new_data_d_2d[method][1][idx]]

        else:
            data[f"f{idx+1}"][method]=[new_data_d_2d[method][0][idx], new_data_d_2d[method][1][idx]]  # mean std
# data = {method: [data[0][0], data[1][0]] for method, data in new_data_d_2d.items()}

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

# # 数据：每个方法 [均值, 标准差]
# data = index_0_data

for p_num in data.keys():

    f_data=data[p_num]
    methods = list(f_data.keys())


    # 提取均值和标准差
    means = np.array([f_data[k][0] for k in methods])      # 均值
    stds = np.array([f_data[k][1] for k in methods])       # 标准差

    # 颜色映射
    colors = ['#88c4d7', '#d0ead5', '#b5e2e5', '#9793c6', '#e6c7df', '#f79691']
    color_map = dict(zip(methods, colors))

    # 创建主图
    fig, ax = plt.subplots(figsize=(6, 6))

    # 散点图：x=均值, y=标准差
    for i, method in enumerate(methods):
        color = color_map[method]
        ax.scatter(means[i], stds[i], color=color, s=210, edgecolors='black', linewidth=1, label=method, zorder=3)

    # 回归线
    slope, intercept = np.polyfit(means, stds, 1)
    line_x = np.array([means.min() - 0.01, means.max() + 0.01])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color='black', linewidth=2, linestyle='--', alpha=0.8, zorder=2)

    # # 标注方法名（微调避免重叠）
    # for i, method in enumerate(methods):
    #     ax.annotate(method, (means[i], stds[i]),
    #                 xytext=(5, 5), textcoords='offset points', fontsize=10,
    #                 alpha=0.9, color='black', weight='bold')

    ax.annotate('Baseline', (means[0], stds[0]),
                    xytext=(-50, 11), textcoords='offset points', fontsize=12,
                    alpha=1, color='black', weight='bold')

    # 统计信息
    r, p = stats.pearsonr(means, stds)
    r_squared = r**2
    textstr = f'r = {r:.2f}\nR² = {r_squared:.2f}\np < 0.001' if p < 0.001 else f'r = {r:.2f}\nR² = {r_squared:.2f}\np = {p:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.3)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props)

    # 坐标轴标签
    ax.set_xlabel('Mean Values', fontsize=18, fontweight='bold')
    ax.set_ylabel('Standard Deviation Values', fontsize=18, fontweight='bold')
    # ax.set_title('Method Comparison: Mean vs Standard Deviation', fontsize=15, pad=20)

    for b in ['bottom', 'left']:
        ax.spines[b].set_linewidth(2.5)   

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 网格
    ax.grid(True, alpha=0.7, linewidth=2.5, color='#CCCCCC', linestyle='--')

    # 图例
    ax.legend(loc='best', fontsize=15, framealpha=0.7, frameon=True, fancybox=True)

    ax.tick_params(labelsize=14, direction='out', length=4.5, width=1.5)

    # =====================
    # 边缘直方图：使用真实值
    # =====================

    divider = make_axes_locatable(ax)

    # 顶部直方图：显示均值分布（条子高度 = 均值）
    ax_top = divider.append_axes("top", size="15%", pad=0.15)
    bars_top = ax_top.bar(range(len(methods)), means, 
                        color=[color_map[m] for m in methods], 
                        edgecolor='black', linewidth=0.5, alpha=0.8)
    ax_top.set_xticks(range(len(methods)))
    # ax_top.set_xticklabels(methods, rotation=0, ha='center')         
    ax_top.set_xticklabels([])
    ax_top.set_yticks([])    


    # ➕ 添加折线
    x_pos = np.arange(len(methods))
    ax_top.plot(x_pos, means, color='black', linewidth=1)


    # ax_top.set_ylabel('Mean Values', fontsize=10)
    # ax_top.set_title('Distribution of Mean Values', fontsize=11)

    # 右侧直方图：显示标准差分布（条子宽度 = 标准差）
    ax_right = divider.append_axes("right", size="15%", pad=0.15)
    bars_right = ax_right.barh(range(len(methods)), stds, 
                            color=[color_map[m] for m in methods], 
                            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax_right.set_yticks(range(len(methods)))
    # ax_right.set_yticklabels(methods, rotation=0, ha='right')
    ax_right.set_yticklabels([])
    ax_right.set_xticks([])
    # ax_right.set_xlabel('Standard Deviation Values', fontsize=10)
    # ax_right.set_title('Distribution of Std Dev Values', fontsize=11, rotation=270, pad=20)

    # ➕ 添加折线
    y_pos = np.arange(len(methods))
    ax_right.plot(stds, y_pos, color='black', linewidth=1)



    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.spines['bottom'].set_visible(True)
    ax_top.spines['bottom'].set_linewidth(1.5)
    ax_top.tick_params(direction='out', length=3.5, width=1)

    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(True)
    ax_right.spines['bottom'].set_visible(False)  
    ax_right.spines['left'].set_linewidth(1.5)
    ax_right.tick_params(direction='out', length=3.5, width=1)

    # 调整布局
    plt.tight_layout()
    for mod in ['pdf', 'svg', 'png']:
        plt.savefig(f"figure/{p_num}.{mod}", dpi=600)

    # plt.show()


