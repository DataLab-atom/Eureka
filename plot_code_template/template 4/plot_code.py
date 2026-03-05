

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["figure.titlesize"] = 15

layer_names = ["layer1", "layer2", "layer3", "layer4"]
num_layers = len(layer_names)

x = np.arange(num_layers) 

ours_mean = np.array([0.1, 0.15, 0.2, 0.18])
ours_std = np.array([0.01, 0.015, 0.02, 0.018])

dimon_mean = np.array([0.12, 0.14, 0.22, 0.19])
dimon_std = np.array([0.012, 0.014, 0.022, 0.019])

fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor='w')

color_ours = '#88c4d7'  
color_dimon = '#f79691'  

ax.plot(x, ours_mean, label='Ours', color=color_ours, linewidth=2, marker='o')
ax.fill_between(x, ours_mean - ours_std, ours_mean + ours_std, color=color_ours, alpha=0.2)

ax.plot(x, dimon_mean, label='DIMON', color=color_dimon, linewidth=2, marker='s')
ax.fill_between(x, dimon_mean - dimon_std, dimon_mean + dimon_std, color=color_dimon, alpha=0.2)

ax.set_xticks(x)
ax.set_xticklabels(layer_names, fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(),fontsize=16)
ax.set_xlabel('Network Layers', fontsize=18)
ax.set_ylabel('Gradient Noise', fontsize=18)
ax.set_ylim(0.05, max(np.max(dimon_mean + dimon_std), np.max(ours_mean + ours_std)) * 1.15)


plt.grid(True, which='major', linestyle='--', alpha=0.7)

ax.legend(loc='upper right', frameon=False,ncol=2,fontsize=16)


fig.tight_layout()

save_path = 'charts\\chart_000.png'
plt.savefig(
    'charts\\chart_000.pdf', 
    dpi=300,             
    bbox_inches='tight', 
    pad_inches=0         
)

plt.show()
