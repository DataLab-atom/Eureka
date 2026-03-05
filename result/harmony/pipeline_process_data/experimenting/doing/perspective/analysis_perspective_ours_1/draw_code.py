import pandas as pd


# 设置归一化热力图的值
min_data=10
max_data=0
for method_name in ['harmony', 'ours', 'origin']:
    data_1 = pd.read_pickle(f'{method_name}.pkl')
    data_1.head()
    data_1 = data_1[['UMAP 1','UMAP 2']]
    data_2 = pd.read_csv(f'{method_name}.csv')
    data_2.head()
    data_3 = pd.read_csv('tma_both_cleaned_meta.csv')
    data_3.head()
    donors,celltypes =  sorted(list(set(data_3['donor'].tolist()))),sorted(list(set(data_3['celltype'].tolist())))
    donors,celltypes
    data_full = {}
    data_2d = {}
    for donor in donors:
        data_full[donor] = {}
        data_2d[donor] = {}
        for celltype in celltypes:
            data_full[donor][celltype] = {}
            data_2d[donor][celltype] = {}

    for donor in donors:
        for celltype in celltypes:
            data_full[donor][celltype] = data_2[data_3['celltype'] == celltype].to_numpy()
            data_2d[donor][celltype] = data_1[(data_3['donor'] == donor) & (data_3['celltype'] == celltype)].to_numpy()
    data_full[donors[0]][celltypes[0]].shape,data_full[donors[1]][celltypes[0]].shape,data_2d[donors[0]][celltypes[0]].shape,data_2d[donors[1]][celltypes[0]].shape
    data_2d[donors[0]][celltypes[0]],data_2d[donors[1]][celltypes[0]]

    import numpy as np
    from sklearn.metrics.pairwise import cosine_distances
    celltypes_len = len(celltypes)
    hot_plot_data = np.zeros((celltypes_len,celltypes_len))
    for i in range(celltypes_len):
        for j in range(celltypes_len):
            # 计算协方差矩阵（rowvar=False 表示列是特征）
            cov1 = np.cov(data_full[donors[0]][celltypes[i]], rowvar=False)  # 形状：(30, 30)
            cov2 = np.cov(data_full[donors[1]][celltypes[j]], rowvar=False)  # 形状：(30, 30)

            # 计算 余弦 距离
            cosine_dist = cosine_distances(cov1, cov2)[0][0]#  np.linalg.norm( - , ord='fro')
            hot_plot_data[i][j] = cosine_dist

    if np.nanmin(hot_plot_data)<min_data:
        min_data=np.nanmin(hot_plot_data)

    if np.nanmax(hot_plot_data)>max_data:
        max_data=np.nanmax(hot_plot_data)


import matplotlib.pyplot as plt
norm = plt.Normalize(min_data, max_data)


# harmony

data_1 = pd.read_pickle('harmony.pkl')
data_1.head()
data_1 = data_1[['UMAP 1','UMAP 2']]
data_2 = pd.read_csv('harmony.csv')
data_2.head()
data_3 = pd.read_csv('tma_both_cleaned_meta.csv')
data_3.head()
donors,celltypes =  sorted(list(set(data_3['donor'].tolist()))),sorted(list(set(data_3['celltype'].tolist())))
donors,celltypes
data_full = {}
data_2d = {}
for donor in donors:
    data_full[donor] = {}
    data_2d[donor] = {}
    for celltype in celltypes:
        data_full[donor][celltype] = {}
        data_2d[donor][celltype] = {}

for donor in donors:
    for celltype in celltypes:
        data_full[donor][celltype] = data_2[data_3['celltype'] == celltype].to_numpy()
        data_2d[donor][celltype] = data_1[(data_3['donor'] == donor) & (data_3['celltype'] == celltype)].to_numpy()
data_full[donors[0]][celltypes[0]].shape,data_full[donors[1]][celltypes[0]].shape,data_2d[donors[0]][celltypes[0]].shape,data_2d[donors[1]][celltypes[0]].shape
data_2d[donors[0]][celltypes[0]],data_2d[donors[1]][celltypes[0]]

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
celltypes_len = len(celltypes)
hot_plot_data = np.zeros((celltypes_len,celltypes_len))
for i in range(celltypes_len):
    for j in range(celltypes_len):
        # 计算协方差矩阵（rowvar=False 表示列是特征）
        cov1 = np.cov(data_full[donors[0]][celltypes[i]], rowvar=False)  # 形状：(30, 30)
        cov2 = np.cov(data_full[donors[1]][celltypes[j]], rowvar=False)  # 形状：(30, 30)

        # 计算 余弦 距离
        cosine_dist = cosine_distances(cov1, cov2)[0][0]#  np.linalg.norm( - , ord='fro')
        hot_plot_data[i][j] = cosine_dist

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines

plt.rcParams['font.family'] = 'Arial'

# ----------------------
# 1. 准备数据（以12×12矩阵为例）
# ----------------------
np.random.seed(42)
matrix_size = 12

# 🆕 替换为字母标签 A, B, C, ...
cell_labels = [chr(ord('A') + i) for i in range(matrix_size)]  # A, B, C, ..., L

# ----------------------
# 2. 绘制基础热力图（隐藏右上三角的颜色块，避免遮挡子图）
# ----------------------
plt.style.use('seaborn-v0_8-white')
fig, ax = plt.subplots(figsize=(12, 10))

mask = np.triu(np.ones_like(hot_plot_data, dtype=bool))
sns.heatmap(
    hot_plot_data,
    annot=True,
    fmt='.2f',
    cmap='viridis',
    annot_kws={"size": 12.5},
    square=True,
    cbar=False,
    linewidths=0.5,
    linecolor='gray',
    mask=mask,
    ax=ax
)


# ✅ 新增：在右侧添加 colorbar
cbar_ax = fig.add_axes([0.765, 0.078, 0.03, 0.82])  # [left, bottom, width, height] —— 右侧边缘

sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
sm.set_array([])

cb = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')

# 👇 移除 colorbar 标题
cb.set_label('')  # ← 空字符串

# 设置刻度和标签
tick_locs = np.linspace(min_data, max_data, 6)
cb.set_ticks(tick_locs)
cb.set_ticklabels([f'{x:.2f}' for x in tick_locs])
cb.ax.tick_params(labelsize=13)


# ----------------------
# 3. 定义右上三角单元格位置计算函数（不变）
# ----------------------
def get_cell_bounds(i, j, matrix_size):
    xmin, xmax = j, j + 1
    ymin, ymax = (matrix_size - 1 - i), (matrix_size - i)
    return xmin, xmax, ymin, ymax


# ----------------------
# 4. 在右上三角每个单元格嵌入子图（不变）
# ----------------------
for i in range(matrix_size):
    for j in range(matrix_size):
        if i-1 < j:  # 右上三角区域
            xmin, xmax, ymin, ymax = get_cell_bounds(11 - i, j, matrix_size)
            inset_ax = inset_axes(
                ax,
                width="80%",
                height="80%",
                loc='center',
                bbox_to_anchor=(xmin, ymin, xmax - xmin, ymax - ymin),
                bbox_transform=ax.transData
            )

            # KDE 绘图（不变）
            sns.kdeplot(
                x=data_2d[donors[0]][celltypes[i]][:,0],
                y=data_2d[donors[0]][celltypes[i]][:,1],
                ax=inset_ax,
                fill=True,
                levels=2,
                color="#cfff36",
                alpha=0.5,
                linewidths=0,
                cut=3
            )
            sns.kdeplot(
                x=data_2d[donors[1]][celltypes[j]][:,0],
                y=data_2d[donors[1]][celltypes[j]][:,1],
                ax=inset_ax,
                fill=True,
                levels=2,
                color="#2c6f86",
                alpha=0.7,
                linewidths=0,
                cut=3
            )

            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_aspect('auto', adjustable='box')


# ----------------------
# 5. 创建图例（放在右侧，上下排列）
# ----------------------

# Batch 图例（红色/蓝色方块）
red_dot = mlines.Line2D([], [], color="#cfff36", marker='s', linestyle='None', markersize=12, alpha=0.5, label='10x')
blue_dot = mlines.Line2D([], [], color="#2c6f86", marker='s', linestyle='None', markersize=12, alpha=0.7, label='SS2')

# Cell Types 图例（字母对应名称）
letter_handles = []
for i, (letter, name) in enumerate(zip(cell_labels, celltypes)):
    letter_handles.append(mlines.Line2D([], [], color='w', label=f'{letter}={name}'))

# ✅ 将两个图例都放在右侧，上下排列（使用两个 axes）
# 上方：Batch 图例
batch_legend_ax = fig.add_axes([0.87, 0.72, 0.08, 0.25])  # [left, bottom, width, height]
batch_legend_ax.axis('off')
batch_legend = batch_legend_ax.legend(handles=[red_dot, blue_dot],
                                      loc='center',
                                      fontsize=13,
                                      title="Batch",
                                      title_fontsize=15,
                                      frameon=False,
                                      edgecolor='black',
                                      labelspacing=1.4,
                                      ncol=1)
# batch_legend.get_frame().set_linewidth(1.5)
batch_legend.get_title().set_fontweight('bold')

# 下方：Cell Types 图例（2行 × 6列）
celltype_legend_ax = fig.add_axes([0.86, 0.1, 0.08, 0.82])  # [left, bottom, width, height]
celltype_legend_ax.axis('off')
# 删除原来的 celltype_legend = celltype_legend_ax.legend(...) 那部分

# ✅ 手动绘制 Cell Types 图例
celltype_legend_ax.axis('off')
y_start = 0.63  # 起始 y 位置（顶部）
dy = 0.055       # 每行间距（可调！越大越稀疏）

for i, (letter, name) in enumerate(zip(cell_labels, celltypes)):
    y_pos = y_start - i * dy
    celltype_legend_ax.text(
        0.08, y_pos, f'{letter}={name}',
        fontsize=13,
        verticalalignment='center',
        horizontalalignment='left',
        transform=celltype_legend_ax.transAxes
    )

# 添加标题
celltype_legend_ax.text(
    0.78, 0.69, "Cell Types",
    fontsize=15, fontweight='bold',
    verticalalignment='center',
    horizontalalignment='center',
    transform=celltype_legend_ax.transAxes
)
# celltype_legend.get_frame().set_linewidth(1.5)


# ----------------------
# 6. 美化坐标轴标签（不变）
# ----------------------
ax.set_xticklabels(cell_labels, rotation=0, ha='right', fontsize=14)
ax.set_yticklabels(cell_labels, rotation=0, fontsize=14)

# 主标题
fig.suptitle('Dataset mixing and cell dissimilarity across 52,000 cells analyzed with Harmony', fontsize=18, fontweight='bold', y=0.975, x=0.5)

# 调整布局，避免右侧图例和 colorbar 重叠
# plt.tight_layout(rect=[0, 0, 0.75, 1.05])  # 左边留白，右边留给图例和 colorbar # 
plt.tight_layout(rect=[0.015, -0.01, 0.75, 1]) 

fig.savefig('charts/harmony.png', dpi=600)



# ours


data_1 = pd.read_pickle('ours.pkl')
data_1.head()
data_1 = data_1[['UMAP 1','UMAP 2']]
data_2 = pd.read_csv('ours.csv')
data_2.head()
data_3 = pd.read_csv('tma_both_cleaned_meta.csv')
data_3.head()
donors,celltypes =  sorted(list(set(data_3['donor'].tolist()))),sorted(list(set(data_3['celltype'].tolist())))
donors,celltypes
data_full = {}
data_2d = {}
for donor in donors:
    data_full[donor] = {}
    data_2d[donor] = {}
    for celltype in celltypes:
        data_full[donor][celltype] = {}
        data_2d[donor][celltype] = {}

for donor in donors:
    for celltype in celltypes:
        data_full[donor][celltype] = data_2[data_3['celltype'] == celltype].to_numpy()
        data_2d[donor][celltype] = data_1[(data_3['donor'] == donor) & (data_3['celltype'] == celltype)].to_numpy()
data_full[donors[0]][celltypes[0]].shape,data_full[donors[1]][celltypes[0]].shape,data_2d[donors[0]][celltypes[0]].shape,data_2d[donors[1]][celltypes[0]].shape
data_2d[donors[0]][celltypes[0]],data_2d[donors[1]][celltypes[0]]

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
celltypes_len = len(celltypes)
hot_plot_data = np.zeros((celltypes_len,celltypes_len))
for i in range(celltypes_len):
    for j in range(celltypes_len):
        # 计算协方差矩阵（rowvar=False 表示列是特征）
        cov1 = np.cov(data_full[donors[0]][celltypes[i]], rowvar=False)  # 形状：(30, 30)
        cov2 = np.cov(data_full[donors[1]][celltypes[j]], rowvar=False)  # 形状：(30, 30)

        # 计算 余弦 距离
        cosine_dist = cosine_distances(cov1, cov2)[0][0]#  np.linalg.norm( - , ord='fro')
        hot_plot_data[i][j] = cosine_dist



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines

plt.rcParams['font.family'] = 'Arial'

# ----------------------
# 1. 准备数据（以12×12矩阵为例）
# ----------------------
np.random.seed(42)
matrix_size = 12

# 🆕 替换为字母标签 A, B, C, ...
cell_labels = [chr(ord('A') + i) for i in range(matrix_size)]  # A, B, C, ..., L

# ----------------------
# 2. 绘制基础热力图（隐藏右上三角的颜色块，避免遮挡子图）
# ----------------------
plt.style.use('seaborn-v0_8-white')
fig, ax = plt.subplots(figsize=(12, 10))

mask = np.triu(np.ones_like(hot_plot_data, dtype=bool))
sns.heatmap(
    hot_plot_data,
    annot=True,
    fmt='.2f',
    cmap='viridis',
    annot_kws={"size": 12.5},
    square=True,
    cbar=False,
    linewidths=0.5,
    linecolor='gray',
    mask=mask,
    ax=ax
)


# ✅ 新增：在右侧添加 colorbar
cbar_ax = fig.add_axes([0.765, 0.078, 0.03, 0.82])  # [left, bottom, width, height] —— 右侧边缘

sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
sm.set_array([])

cb = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')

# 👇 移除 colorbar 标题
cb.set_label('')  # ← 空字符串

# 设置刻度和标签
tick_locs = np.linspace(min_data, max_data, 6)
cb.set_ticks(tick_locs)
cb.set_ticklabels([f'{x:.2f}' for x in tick_locs])
cb.ax.tick_params(labelsize=13)


# ----------------------
# 3. 定义右上三角单元格位置计算函数（不变）
# ----------------------
def get_cell_bounds(i, j, matrix_size):
    xmin, xmax = j, j + 1
    ymin, ymax = (matrix_size - 1 - i), (matrix_size - i)
    return xmin, xmax, ymin, ymax


# ----------------------
# 4. 在右上三角每个单元格嵌入子图（不变）
# ----------------------
for i in range(matrix_size):
    for j in range(matrix_size):
        if i-1 < j:  # 右上三角区域
            xmin, xmax, ymin, ymax = get_cell_bounds(11 - i, j, matrix_size)
            inset_ax = inset_axes(
                ax,
                width="80%",
                height="80%",
                loc='center',
                bbox_to_anchor=(xmin, ymin, xmax - xmin, ymax - ymin),
                bbox_transform=ax.transData
            )

            # KDE 绘图（不变）
            sns.kdeplot(
                x=data_2d[donors[0]][celltypes[i]][:,0],
                y=data_2d[donors[0]][celltypes[i]][:,1],
                ax=inset_ax,
                fill=True,
                levels=2,
                color="#cfff36",
                alpha=0.5,
                linewidths=0,
                cut=3
            )
            sns.kdeplot(
                x=data_2d[donors[1]][celltypes[j]][:,0],
                y=data_2d[donors[1]][celltypes[j]][:,1],
                ax=inset_ax,
                fill=True,
                levels=2,
                color="#2c6f86",
                alpha=0.7,
                linewidths=0,
                cut=3
            )

            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_aspect('auto', adjustable='box')


# ----------------------
# 5. 创建图例（放在右侧，上下排列）
# ----------------------

# Batch 图例（红色/蓝色方块）
red_dot = mlines.Line2D([], [], color="#cfff36", marker='s', linestyle='None', markersize=12, alpha=0.5, label='10x')
blue_dot = mlines.Line2D([], [], color="#2c6f86", marker='s', linestyle='None', markersize=12, alpha=0.7, label='SS2')

# Cell Types 图例（字母对应名称）
letter_handles = []
for i, (letter, name) in enumerate(zip(cell_labels, celltypes)):
    letter_handles.append(mlines.Line2D([], [], color='w', label=f'{letter}={name}'))

# ✅ 将两个图例都放在右侧，上下排列（使用两个 axes）
# 上方：Batch 图例
batch_legend_ax = fig.add_axes([0.87, 0.72, 0.08, 0.25])  # [left, bottom, width, height]
batch_legend_ax.axis('off')
batch_legend = batch_legend_ax.legend(handles=[red_dot, blue_dot],
                                      loc='center',
                                      fontsize=13,
                                      title="Batch",
                                      title_fontsize=15,
                                      frameon=False,
                                      edgecolor='black',
                                      labelspacing=1.4,
                                      ncol=1)
# batch_legend.get_frame().set_linewidth(1.5)
batch_legend.get_title().set_fontweight('bold')

# 下方：Cell Types 图例（2行 × 6列）
celltype_legend_ax = fig.add_axes([0.86, 0.1, 0.08, 0.82])  # [left, bottom, width, height]
celltype_legend_ax.axis('off')
# 删除原来的 celltype_legend = celltype_legend_ax.legend(...) 那部分

# ✅ 手动绘制 Cell Types 图例
celltype_legend_ax.axis('off')
y_start = 0.63  # 起始 y 位置（顶部）
dy = 0.055       # 每行间距（可调！越大越稀疏）

for i, (letter, name) in enumerate(zip(cell_labels, celltypes)):
    y_pos = y_start - i * dy
    celltype_legend_ax.text(
        0.08, y_pos, f'{letter}={name}',
        fontsize=13,
        verticalalignment='center',
        horizontalalignment='left',
        transform=celltype_legend_ax.transAxes
    )

# 添加标题
celltype_legend_ax.text(
    0.78, 0.69, "Cell Types",
    fontsize=15, fontweight='bold',
    verticalalignment='center',
    horizontalalignment='center',
    transform=celltype_legend_ax.transAxes
)
# celltype_legend.get_frame().set_linewidth(1.5)


# ----------------------
# 6. 美化坐标轴标签（不变）
# ----------------------
ax.set_xticklabels(cell_labels, rotation=0, ha='right', fontsize=14)
ax.set_yticklabels(cell_labels, rotation=0, fontsize=14)

# 主标题
fig.suptitle('Dataset mixing and cell dissimilarity across 52,000 cells analyzed with MoE-Harmony', fontsize=18, fontweight='bold', y=0.975, x=0.5)

# 调整布局，避免右侧图例和 colorbar 重叠
# plt.tight_layout(rect=[0, 0, 0.75, 1.05])  # 左边留白，右边留给图例和 colorbar # 
plt.tight_layout(rect=[0.015, -0.01, 0.75, 1]) 

# 保存图片：png
fig.savefig('charts/ours.png', dpi=600)


# origin


data_1 = pd.read_pickle('origin.pkl')
data_1.head()
data_1 = data_1[['UMAP 1','UMAP 2']]
data_2 = pd.read_csv('origin.csv')
data_2.head()
data_3 = pd.read_csv('tma_both_cleaned_meta.csv')
data_3.head()
donors,celltypes =  sorted(list(set(data_3['donor'].tolist()))),sorted(list(set(data_3['celltype'].tolist())))
donors,celltypes
data_full = {}
data_2d = {}
for donor in donors:
    data_full[donor] = {}
    data_2d[donor] = {}
    for celltype in celltypes:
        data_full[donor][celltype] = {}
        data_2d[donor][celltype] = {}

for donor in donors:
    for celltype in celltypes:
        data_full[donor][celltype] = data_2[data_3['celltype'] == celltype].to_numpy()
        data_2d[donor][celltype] = data_1[(data_3['donor'] == donor) & (data_3['celltype'] == celltype)].to_numpy()
data_full[donors[0]][celltypes[0]].shape,data_full[donors[1]][celltypes[0]].shape,data_2d[donors[0]][celltypes[0]].shape,data_2d[donors[1]][celltypes[0]].shape
data_2d[donors[0]][celltypes[0]],data_2d[donors[1]][celltypes[0]]

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
celltypes_len = len(celltypes)
hot_plot_data = np.zeros((celltypes_len,celltypes_len))
for i in range(celltypes_len):
    for j in range(celltypes_len):
        # 计算协方差矩阵（rowvar=False 表示列是特征）
        cov1 = np.cov(data_full[donors[0]][celltypes[i]], rowvar=False)  # 形状：(30, 30)
        cov2 = np.cov(data_full[donors[1]][celltypes[j]], rowvar=False)  # 形状：(30, 30)

        # 计算 余弦 距离
        cosine_dist = cosine_distances(cov1, cov2)[0][0]#  np.linalg.norm( - , ord='fro')
        hot_plot_data[i][j] = cosine_dist


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines

plt.rcParams['font.family'] = 'Arial'

# ----------------------
# 1. 准备数据（以12×12矩阵为例）
# ----------------------
np.random.seed(42)
matrix_size = 12

# 🆕 替换为字母标签 A, B, C, ...
cell_labels = [chr(ord('A') + i) for i in range(matrix_size)]  # A, B, C, ..., L

# ----------------------
# 2. 绘制基础热力图（隐藏右上三角的颜色块，避免遮挡子图）
# ----------------------
plt.style.use('seaborn-v0_8-white')
fig, ax = plt.subplots(figsize=(12, 10))

mask = np.triu(np.ones_like(hot_plot_data, dtype=bool))
sns.heatmap(
    hot_plot_data,
    annot=True,
    fmt='.2f',
    cmap='viridis',
    annot_kws={"size": 12.5},
    square=True,
    cbar=False,
    linewidths=0.5,
    linecolor='gray',
    mask=mask,
    ax=ax
)


# ✅ 新增：在右侧添加 colorbar
cbar_ax = fig.add_axes([0.765, 0.078, 0.03, 0.82])  # [left, bottom, width, height] —— 右侧边缘

sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
sm.set_array([])

cb = plt.colorbar(sm, cax=cbar_ax, orientation='vertical')

# 👇 移除 colorbar 标题
cb.set_label('')  # ← 空字符串

# 设置刻度和标签
tick_locs = np.linspace(min_data, max_data, 6)
cb.set_ticks(tick_locs)
cb.set_ticklabels([f'{x:.2f}' for x in tick_locs])
cb.ax.tick_params(labelsize=13)


# ----------------------
# 3. 定义右上三角单元格位置计算函数（不变）
# ----------------------
def get_cell_bounds(i, j, matrix_size):
    xmin, xmax = j, j + 1
    ymin, ymax = (matrix_size - 1 - i), (matrix_size - i)
    return xmin, xmax, ymin, ymax


# ----------------------
# 4. 在右上三角每个单元格嵌入子图（不变）
# ----------------------
for i in range(matrix_size):
    for j in range(matrix_size):
        if i-1 < j:  # 右上三角区域
            xmin, xmax, ymin, ymax = get_cell_bounds(11 - i, j, matrix_size)
            inset_ax = inset_axes(
                ax,
                width="80%",
                height="80%",
                loc='center',
                bbox_to_anchor=(xmin, ymin, xmax - xmin, ymax - ymin),
                bbox_transform=ax.transData
            )

            # KDE 绘图（不变）
            sns.kdeplot(
                x=data_2d[donors[0]][celltypes[i]][:,0],
                y=data_2d[donors[0]][celltypes[i]][:,1],
                ax=inset_ax,
                fill=True,
                levels=2,
                color="#cfff36",
                alpha=0.5,
                linewidths=0,
                cut=3
            )
            sns.kdeplot(
                x=data_2d[donors[1]][celltypes[j]][:,0],
                y=data_2d[donors[1]][celltypes[j]][:,1],
                ax=inset_ax,
                fill=True,
                levels=2,
                color="#2c6f86",
                alpha=0.7,
                linewidths=0,
                cut=3
            )

            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_aspect('auto', adjustable='box')


# ----------------------
# 5. 创建图例（放在右侧，上下排列）
# ----------------------

# Batch 图例（红色/蓝色方块）
red_dot = mlines.Line2D([], [], color="#cfff36", marker='s', linestyle='None', markersize=12, alpha=0.5, label='10x')
blue_dot = mlines.Line2D([], [], color="#2c6f86", marker='s', linestyle='None', markersize=12, alpha=0.7, label='SS2')

# Cell Types 图例（字母对应名称）
letter_handles = []
for i, (letter, name) in enumerate(zip(cell_labels, celltypes)):
    letter_handles.append(mlines.Line2D([], [], color='w', label=f'{letter}={name}'))

# ✅ 将两个图例都放在右侧，上下排列（使用两个 axes）
# 上方：Batch 图例
batch_legend_ax = fig.add_axes([0.87, 0.72, 0.08, 0.25])  # [left, bottom, width, height]
batch_legend_ax.axis('off')
batch_legend = batch_legend_ax.legend(handles=[red_dot, blue_dot],
                                      loc='center',
                                      fontsize=13,
                                      title="Batch",
                                      title_fontsize=15,
                                      frameon=False,
                                      edgecolor='black',
                                      labelspacing=1.4,
                                      ncol=1)
# batch_legend.get_frame().set_linewidth(1.5)
batch_legend.get_title().set_fontweight('bold')

# 下方：Cell Types 图例（2行 × 6列）
celltype_legend_ax = fig.add_axes([0.86, 0.1, 0.08, 0.82])  # [left, bottom, width, height]
celltype_legend_ax.axis('off')
# 删除原来的 celltype_legend = celltype_legend_ax.legend(...) 那部分

# ✅ 手动绘制 Cell Types 图例
celltype_legend_ax.axis('off')
y_start = 0.63  # 起始 y 位置（顶部）
dy = 0.055       # 每行间距（可调！越大越稀疏）

for i, (letter, name) in enumerate(zip(cell_labels, celltypes)):
    y_pos = y_start - i * dy
    celltype_legend_ax.text(
        0.08, y_pos, f'{letter}={name}',
        fontsize=13,
        verticalalignment='center',
        horizontalalignment='left',
        transform=celltype_legend_ax.transAxes
    )

# 添加标题
celltype_legend_ax.text(
    0.78, 0.69, "Cell Types",
    fontsize=15, fontweight='bold',
    verticalalignment='center',
    horizontalalignment='center',
    transform=celltype_legend_ax.transAxes
)
# celltype_legend.get_frame().set_linewidth(1.5)


# ----------------------
# 6. 美化坐标轴标签（不变）
# ----------------------
ax.set_xticklabels(cell_labels, rotation=0, ha='right', fontsize=14)
ax.set_yticklabels(cell_labels, rotation=0, fontsize=14)

# 主标题
fig.suptitle('Initial value for dataset mixing and cell dissimilarity across 52,000 cells', fontsize=18, fontweight='bold', y=0.975, x=0.5)


# 调整布局，避免右侧图例和 colorbar 重叠
# plt.tight_layout(rect=[0, 0, 0.75, 1.05])  # 左边留白，右边留给图例和 colorbar # 
plt.tight_layout(rect=[0.015, -0.01, 0.75, 1]) 


# plt.tight_layout()
# 保存图片：png
fig.savefig('charts/origin.png', dpi=600)
