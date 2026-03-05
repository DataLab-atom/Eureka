import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12

colors = ["#88c4d7", "#b5e2e5","#f79691"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# 文件路径配置
data_path = ".\\draw_data"
save_path = ".\\charts"

os.makedirs(save_path, exist_ok=True)


np.random.seed(100)  
num_layers = 3

subnetworks = ["Branch1", "Branch2", "Trunk"]
methods = ["Baseline", "Residual"]
robustness_metrics = ["Convergence Error", "Error Variance"]

corr_matrices = dict()
for method in methods:
    corr_matrices[method] = dict()
    for subnet in subnetworks:
        mats = []
        for _ in range(num_layers):
            A = np.random.rand(3,3)
            mat = np.dot(A, A.T)  
            np.fill_diagonal(mat, 1.0)  
            mat = (mat - mat.min()) / (mat.max() - mat.min()) * 2 - 1
            mats.append(mat)
        corr_matrices[method][subnet] = np.array(mats)

robustness_data = dict()
for metric in robustness_metrics:
    robustness_data[metric] = dict()
    for method in methods:
        robustness_data[metric][method] = np.abs(np.random.randn(num_layers, len(subnetworks))) * 0.1 + 0.1


for layer_idx in range(num_layers):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10,6), gridspec_kw={'width_ratios':[0.9,0.9,0.9,1.3], 'height_ratios':[1,1]})
    fig.subplots_adjust(wspace=0.02, hspace=0.02)


    for j, method in enumerate(methods):  
        for i, subnet in enumerate(subnetworks):
            ax = axes[j, i]
            data = corr_matrices[method][subnet][layer_idx]
            sns.heatmap(
                data,
                cmap=custom_cmap,
                vmin=-1, vmax=1,
                annot=True,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                linecolor='white',
                cbar=False,  
                cbar_kws={"shrink":0.6, "label":"Correlation Coefficient"} if (i==2 and j==0) else None,
                ax=ax
            )
            fontsize = 12
            ax.set_xlabel(f"{method}", fontsize=18)
            ax.set_ylabel(f"{subnet}", fontsize=18)
            ax.set_xticklabels(["F1", "F2", "F3"], rotation=0, fontsize=14)
            ax.set_yticklabels(["F1", "F2", "F3"], rotation=0, fontsize=14)

    ax_robust_conv = axes[0, 3]
    bar_width = 0.35
    index = np.arange(len(subnetworks))

    baseline_vals = robustness_data[robustness_metrics[0]][methods[0]][layer_idx]
    residual_vals = robustness_data[robustness_metrics[0]][methods[1]][layer_idx]
    max_val = max(np.max(baseline_vals), np.max(residual_vals))
    bars1 = ax_robust_conv.bar(index, baseline_vals, bar_width, label='DIMON', color='#88c4d7')
    bars2 = ax_robust_conv.bar(index + bar_width, residual_vals, bar_width, label='Ours', color='#9793c6')
    ax_robust_conv.set_title(" \n \n", weight='bold')
    ax_robust_conv.set_ylabel(robustness_metrics[0] , weight='bold')
    ax_robust_conv.set_xticks(index + bar_width / 2)
    ax_robust_conv.set_xticklabels(subnetworks)
    ax_robust_conv.set_ylim(0, max_val * 1.4)
    ax_robust_conv.legend(fontsize=10,ncol=2)
    ax_robust_conv.grid(axis='y', linestyle='--', alpha=0.7)

    ax_robust_var = axes[1, 3]

    baseline_vals_var = robustness_data[robustness_metrics[1]][methods[0]][layer_idx]
    residual_vals_var = robustness_data[robustness_metrics[1]][methods[1]][layer_idx]
    max_val = max(np.max(baseline_vals_var), np.max(residual_vals_var))
    bars3 = ax_robust_var.bar(index, baseline_vals_var, bar_width, label='DIMON', color='#88c4d7')
    bars4 = ax_robust_var.bar(index + bar_width, residual_vals_var, bar_width, label='Ours', color='#9793c6')
    ax_robust_var.set_title(" \n \n", weight='bold')
    ax_robust_var.set_ylabel(robustness_metrics[1] , weight='bold')
    ax_robust_var.set_xticks(index + bar_width / 2)
    ax_robust_var.set_xticklabels(subnetworks)
    ax_robust_var.set_ylim(0, max_val * 1.4)
    ax_robust_var.legend(fontsize=10,ncol=2)
    ax_robust_var.grid(axis='y', linestyle='--', alpha=0.7)


    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filename = os.path.join(save_path, f'chart_{layer_idx:03d}.pdf')
    plt.savefig(
    filename,  
    dpi=300,            
    bbox_inches='tight',
    pad_inches=0)       
    plt.close(fig)

