import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

# Set font globally for professional academic look
plt.rcParams['font.family'] = 'Arial'

# Paths
data_path = "./experimental_result_data/responsibility_entropy_correction_per_cell.csv"
spearman_path = "./experimental_result_data/spearman_summary.csv"
output_dir = "./charts"
os.makedirs(output_dir, exist_ok=True)

# Load data
plot_df = pd.read_csv(data_path)
spearman_df = pd.read_csv(spearman_path)

# Define methods and titles for panels
methods = ['MoE-Harmony', 'Harmony']
method_titles = {'MoE-Harmony': 'MoE-Harmony', 'Harmony': 'Harmony'}

# Create dictionary for spearman correlation and p-values
spearman_stats = {row['method']: (row['spearman_rho'], row['p_value']) for _, row in spearman_df.iterrows()}

# Custom colormap for better visual appeal
custom_colors = ["#2c6f86", "#57b35f", "#cfff36", "#fff800"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)

# Set seaborn theme for clean academic style
sns.set_theme(style="ticks", font_scale=1.1)

# Create figure with two side-by-side panels
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(1, 2, wspace=0.15)

# Determine unified y-axis limit for correction magnitude
y_max = plot_df['correction_magnitude'].max()
y_lim = (0, y_max * 1.1)

for i, method in enumerate(methods):
    ax = fig.add_subplot(gs[i])
    df_sub = plot_df[plot_df['method'] == method]

    # KDE plot with fill for smooth density visualization
    kde = sns.kdeplot(
        data=df_sub,
        x='normalized_entropy',
        y='correction_magnitude',
        fill=True,
        levels=10,  # smooth contour levels
        cmap=custom_cmap,
        thresh=0.05,  # ignore very low density background
        cut=0,
        bw_adjust=1.2,
        ax=ax,
        cbar=True,
        cbar_kws={'label': 'Density', 'shrink': 0.8, 'format': '%.2f'}
    )

    # Customize colorbar label and ticks
    cbar = ax.collections[-1].colorbar
    cbar.set_label('Density', fontsize=17, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=11)

    # Set axis limits
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_lim)

    # Axis labels and title
    ax.set_xlabel('Normalized Entropy', fontsize=17, fontweight='bold')
    if i == 0:
        ax.set_ylabel('Correction Magnitude', fontsize=17, fontweight='bold')
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False, labelsize=11)

    ax.set_title(method_titles[method], fontsize=21, fontweight='bold', pad=15)

    # Add Spearman correlation annotation
    rho, pval = spearman_stats[method]
    p_text = f"{pval:.1e}" if pval < 0.001 else f"{pval:.3f}"
    annotation_text = rf'$h$ = {rho:.3f}' + '\n' + rf'$p$ = {p_text}'

    ax.text(
        0.7, 0.08, annotation_text,
        ha='left', va='center', transform=ax.transAxes,
        fontsize=13,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.5')
    )

    # Grid for readability
    ax.grid(True, linestyle='--', alpha=0.8)

# Tight layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])

save_file = os.path.join(output_dir, 'responsibility_entropy_vs_correction_smooth_custom.png')
fig.savefig(save_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Chart saved to {save_file}")





























