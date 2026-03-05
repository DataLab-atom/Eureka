import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

plt.rcParams['font.family'] = 'Arial'

# --- Configuration ---
MANUAL_Z_LIMITS = (1.0, 5.0)  # Set fixed Z axis limits for color normalization

# File paths
binned_harmony_fp = r"./experimental_result_data/binned_grid_Harmony.csv"
binned_moe_fp = r"./experimental_result_data/binned_grid_MoE-Harmony.csv"
summary_fp = r"./experimental_result_data/correlation_summary.csv"
out_fp = os.path.join("./charts", "correction_terrain_3d_unified.png")
os.makedirs(os.path.dirname(out_fp), exist_ok=True)

# --- Data loading and processing functions ---
def load_data_points(fp):
    df = pd.read_csv(fp)
    return df['x_center'].values, df['y_center'].values, df['mean_correction'].values

def create_smooth_terrain(x, y, z, resolution=60, sigma=1.2):
    xi = np.linspace(min(x), max(x), resolution)
    yi = np.linspace(min(y), max(y), resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    
    # Fill NaNs with nearest neighbor interpolation
    mask = np.isnan(Zi)
    if mask.any():
        from scipy.interpolate import NearestNDInterpolator
        valid = ~np.isnan(z)
        interpolator = NearestNDInterpolator(list(zip(x[valid], y[valid])), z[valid])
        Zi[mask] = interpolator(Xi[mask], Yi[mask])
    
    # Smooth the surface
    Zi_smooth = gaussian_filter(Zi, sigma=sigma)
    return Xi, Yi, Zi_smooth

# --- Load data ---
hx, hy, hz = load_data_points(binned_harmony_fp)
mx, my, mz = load_data_points(binned_moe_fp)
summary = pd.read_csv(summary_fp)
pearson_map = {row['method']: (row['pearson_r'], row['p_value']) for _, row in summary.iterrows()}

# --- Create smooth terrains ---
X_h, Y_h, Z_h = create_smooth_terrain(hx, hy, hz)
X_m, Y_m, Z_m = create_smooth_terrain(mx, my, mz)

# --- Determine unified Z axis limits and normalization ---
if MANUAL_Z_LIMITS:
    vmin, vmax = MANUAL_Z_LIMITS
else:
    all_z = np.concatenate([Z_h.ravel(), Z_m.ravel()])
    all_z = all_z[~np.isnan(all_z)]
    vmin, vmax = np.min(all_z), np.max(all_z)

norm = Normalize(vmin=vmin, vmax=vmax)

# --- Plotting function ---
def plot_unified_terrain(ax, X, Y, Z, title, stats_key):
    cmap = cm.viridis
    # Plot surface with color normalization
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm, 
                           edgecolor='none', alpha=0.9, rstride=1, cstride=1, 
                           antialiased=False, shade=True)

    # Determine floor offset for contour projection
    if title == 'Harmony':
        z_floor = vmin - (vmax - vmin) * 0.05
    else:
        z_floor = vmin - (vmax - vmin) * 0.6

    # Draw contour projection on bottom plane
    ax.contourf(X, Y, Z, zdir='z', offset=z_floor, cmap=cmap, norm=norm, alpha=0.4)

    # Set Z axis limits
    ax.set_zlim(z_floor, vmax)

    # Axis labels and title
    ax.set_title(title, fontsize=18, y=1.05, weight='bold')
    ax.set_xlabel('\nDiversity', linespacing=2, fontsize=14, weight='bold')
    ax.set_ylabel('\nAlignment', linespacing=2, fontsize=14, weight='bold')
    ax.set_zlabel('\nMean Correction', linespacing=2, fontsize=14, weight='bold')

    # Set viewing angle
    ax.view_init(elev=35, azim=-110)

    # Add correlation stats text box
    r, p = pearson_map.get(stats_key, (np.nan, np.nan))
    ax.text2D(0.3, 0.8, f"r = {r:.3f}\n$p$ = {p:.1e}", transform=ax.transAxes,
              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8), fontsize=12)

    # Make panes transparent and add grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.3)

    return surf

# --- Create figure and axes ---
fig = plt.figure(figsize=(15, 7), constrained_layout=True)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Plot both methods
surf1 = plot_unified_terrain(ax1, X_h, Y_h, Z_h, "Harmony", "Harmony")
surf2 = plot_unified_terrain(ax2, X_m, Y_m, Z_m, "MoE-Harmony", "MoE-Harmony")

# Shared colorbar
sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.7, aspect=20, pad=0.02)
cbar.set_label('Mean Correction Magnitude', fontsize=14, fontweight='bold', labelpad=13)

# Save figure
fig.savefig(out_fp, dpi=300)
print(f"Saved figure to: {out_fp}")
plt.close(fig)

