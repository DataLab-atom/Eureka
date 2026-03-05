# import os
# os.chdir('/data/zz/rag_paper/experiment_auto_write/result/pgrr/angle/analysis_angle_003/draw_data')
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Times New Roman'

# Define the directory paths
base_dir = os.path.dirname(os.path.abspath(__file__))
chart_save_dir = os.path.join(base_dir, 'charts')

# Ensure the chart directory exists
os.makedirs(chart_save_dir, exist_ok=True)

# Data files and labels mapping
file_label_map = {
    'draw_data_000.npy': r'$\mu$=0.0',
    'draw_data_002.npy': r'$\mu$=0.5',
    'draw_data_003.npy': r'$\mu$=0.7',
    'draw_data_004.npy': r'$\mu$=0.9',
    'draw_data_005.npy': r'$\mu$=0.99'
}

# Initialize data container
data_dict = {}

# Load data safely
for filename, label in file_label_map.items():
    data_path = os.path.join(os.path.join(base_dir, 'draw_data'), filename)
    try:
        data = np.load(data_path)
        data_dict[label] = data
    except FileNotFoundError:
        print("的急急急的结局二点")
        pass
        # Instead of failing, create synthetic data with plausible behavior for demonstration
        # This is to ensure code runs without real data files.
        # Synthetic data simulates convergence curves with momentum effect
        # np.random.seed(hash(label) % 123456)
        # iterations = 40  # number of recorded points
        # base_error = np.linspace(1, 0.01, iterations)  # decreasing error
        # noise = np.random.normal(0, 0.02, iterations)  # small noise
        # if label == 'Baseline (momentum=0.0)':
        #     synthetic_data = base_error + noise + 0.05
        # else:
        #     momentum_val = float(label.split('=')[1])
        #     synthetic_data = base_error * (1 - momentum_val * 0.5) + noise
        # synthetic_data = np.clip(synthetic_data, 0.001, None)  # keep positive
        # data_dict[label] = synthetic_data

# Plotting parameters
iteration_interval = 3000
num_points = len(next(iter(data_dict.values())))
iterations = np.arange(0, num_points) * iteration_interval

plt.figure(figsize=(6, 6))

# Define color cycle for clarity
colors = plt.cm.viridis(np.linspace(0, 1, len(data_dict)))
markers=['o', 's', '*', 'v', None]
lss=['-', '-', '-', '-', '-.']

idx=0
for (label, data), color in zip(data_dict.items(), colors):
    plt.plot(iterations, data, label=label, linewidth=4, color=color, marker=markers[idx], markersize=8, ls=lss[idx], alpha=0.95)
    idx+=1

# Axis labels and title
plt.xlabel('Iterations', fontsize=21, fontweight='bold')
plt.ylabel('Normalized Squared Error', fontsize=21, fontweight='bold')
# plt.title('Convergence Curves for Different Momentum Parameters', fontsize=16, fontweight='bold')

# Log scale for y-axis to better show convergence behavior
plt.yscale('log')

# Grid and legend
plt.grid(True, which='both', linestyle='--', linewidth=2, alpha=0.7)
# plt.legend(title='Method', fontsize=11, title_fontsize=13, loc='upper right', framealpha=0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=18, prop={'weight': 'bold'})

# Tight layout for better spacing
plt.tight_layout()

plt.tick_params(axis='both', which='major', labelsize=14, length=5, width=2, direction='out')
plt.tick_params(axis='both', which='minor', labelsize=14)

# 移除上边框和右边框
ax = plt.gca()  # 获取当前坐标轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the figure
# save_path = os.path.join(chart_save_dir, 'chart_000.png')
plt.savefig(os.path.join(chart_save_dir, 'chart_000.png'), dpi=700, bbox_inches='tight')
plt.savefig(os.path.join(chart_save_dir, 'chart_000.svg'), dpi=700, bbox_inches='tight')
plt.savefig(os.path.join(chart_save_dir, 'chart_000.pdf'), dpi=700, bbox_inches='tight')
plt.close()
