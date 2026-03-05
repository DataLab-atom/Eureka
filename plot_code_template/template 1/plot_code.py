
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True


data_dir = ".\\draw_data"
save_dir = ".\\charts"

iterations = np.arange(1, 51)

np.random.seed(42)
baseline_sync_scores = np.clip(0.5 + 0.1 * np.sin(iterations / 3) + 0.05 * np.random.randn(50), 0, 1)
proposed_sync_scores = np.clip(0.6 + 0.1 * np.cos(iterations / 4) + 0.04 * np.random.randn(50), 0, 1)

width = 0.35
x = np.arange(len(iterations))

fig, ax = plt.subplots(figsize=(15, 6), dpi=300, facecolor='w')

bars1 = ax.bar(x - width/2, baseline_sync_scores, width, label='DIMON', color='#88c4d7', edgecolor='k', linewidth=0.8)
bars2 = ax.bar(x + width/2, proposed_sync_scores, width, label='Ours', color='#9793c6', edgecolor='k', linewidth=0.8)

ax.set_xlabel('Training Iteration', fontsize=18)
ax.set_ylabel('Synchronization Score', fontsize=18)

ax.set_xticks(x)
ax.set_xticklabels([str(i) if (i+1)%5==0 else '' for i in x], fontsize=16)
ax.set_ylim(0, 1.1)

ax.legend(frameon=False, fontsize=16,ncol=2)
plt.grid(True, which='major', linestyle='--', alpha=0.7)


plt.tight_layout()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
file_path = os.path.join(save_dir, 'chart_000.png')
plt.savefig(
    '.\\chart_000.pdf', 
    dpi=300,             
    bbox_inches='tight', 
    pad_inches=0         
)
plt.show()