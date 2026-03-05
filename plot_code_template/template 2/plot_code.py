

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["figure.titlesize"] = 15
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.linewidth"] = 1.2

data_dir = ".\\draw_data"
chart_dir = ".\\charts"
if not os.path.exists(chart_dir):
    os.makedirs(chart_dir)

np.random.seed(1234)
num_groups = 4
sample_counts = [80, 100, 60, 50]  

baseline_data = [np.random.normal(loc=1.2 - 0.15*g, scale=0.25, size=count) for g, count in enumerate(sample_counts)]
proposed_data = [np.random.normal(loc=1.0 - 0.12*g, scale=0.18, size=count) for g, count in enumerate(sample_counts)]

colors = ['#b5e2e5', '#e6c7df'] 

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

positions_baseline = np.arange(num_groups)*2.5
positions_proposed = positions_baseline + 0.9
width = 0.75

def draw_boxplot(data, positions, color):
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showmeans=True,
                    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":4})
    for patch in bp['boxes']:
        patch.set(facecolor=color, alpha=0.8, linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(color="#333333", linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color="#333333", linewidth=1.2)
    for median in bp['medians']:
        median.set(color="#111111", linewidth=1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', markersize=4, alpha=0.6, markeredgecolor="#333333", markerfacecolor=color)


draw_boxplot(baseline_data, positions_baseline, colors[0])
draw_boxplot(proposed_data, positions_proposed, colors[1])

mid_positions = (positions_baseline + positions_proposed) / 2
label_groups = ['Bin 0', 'Bin 1', 'Bin 2', 'Bin 3']
ax.set_xticks(mid_positions)
ax.set_xticklabels(label_groups, fontsize=14)

ax.set_xlabel('Label Magnitude Bin', fontsize=15)
ax.set_ylabel('Per-sample Weighted Loss at Final Epoch', fontsize=15)

legend_elements = [Patch(facecolor=colors[0], edgecolor='k', label='DIMON'),
                   Patch(facecolor=colors[1], edgecolor='k', label='Ours')]
ax.legend(handles=legend_elements, fontsize=13, title_fontsize=14, loc='upper right', frameon=False)

ax.grid(axis='y', linestyle='--', alpha=0.5)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

all_losses = np.concatenate(baseline_data + proposed_data)
ax.set_ylim(np.min(all_losses) - 0.1, np.max(all_losses) + 0.1)

save_path = os.path.join(chart_dir, 'chart_000.png')
plt.tight_layout()
plt.savefig(
    '.\\chart_000.pdf',  
    dpi=300,             
    bbox_inches='tight', 
    pad_inches=0         
)
plt.close()
