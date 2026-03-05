import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'

import seaborn as sns
from matplotlib.lines import Line2D

# Statistical imports with safe fallbacks
try:
    from scipy.stats import wilcoxon, ttest_rel, sem, t
except Exception:
    wilcoxon = None
    try:
        from scipy.stats import ttest_rel, sem, t
    except Exception:
        ttest_rel = None
        sem = None
        t = None

# ---------- Config ----------
CSV_PATH = os.path.join('.', 'experimental_result_data', 'mtscf_per_cell.csv')
OUT_DIR = os.path.join('.', 'charts')
OUT_PATH = os.path.join(OUT_DIR, 'mtscf_violin.png')
FIG_DPI = 300

# Figure size for 2x3 grid
FIG_WIDTH = 16
FIG_HEIGHT = 9 

COLOR_MOE = '#57b35f'   # teal
COLOR_H = '#fff800'     # gray
VIOLIN_ALPHA = 0.7      # Transparency
BOX_WIDTH = 0.12
VIOLIN_WIDTH = 0.8

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load data ----------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Required CSV not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_cols = ['cell_id', 'method', 'MTSCF']
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Input CSV missing required column: {c}")

methods_expected = ['MoE_Harmony', 'Harmony']
df = df[df['method'].isin(methods_expected)].copy()
df = df.dropna(subset=['MTSCF'])

df['MTSCF'] = pd.to_numeric(df['MTSCF'], errors='coerce')
df = df.dropna(subset=['MTSCF'])
df['MTSCF'] = df['MTSCF'].clip(0.0, 1.0)

# Facetting
facet_col = 'bio_label' if 'bio_label' in df.columns else None
if facet_col is None:
    df['_facet'] = 'All cells'
    facet_col = '_facet'

# Sort and limit to 6 facets
facets = sorted(df[facet_col].unique())
max_facets = 6 
if len(facets) > max_facets:
    facets = facets[:(max_facets-1)] + ['Other']
    df[facet_col] = df[facet_col].where(df[facet_col].isin(facets), other='Other')

# ---------- Helper statistics ----------
def paired_statistics(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a = a[mask]
    b = b[mask]
    n = len(a)
    mean_a = float(np.nanmean(a)) if n > 0 else np.nan
    mean_b = float(np.nanmean(b)) if n > 0 else np.nan
    pval = np.nan
    stat_name = None
    
    if wilcoxon is not None and n > 0:
        try:
            stat, pval = wilcoxon(a, b)
            stat_name = 'wilcoxon'
        except Exception:
            stat_name = None
            pval = np.nan
    if stat_name is None and ttest_rel is not None and n > 1:
        try:
            stat, pval = ttest_rel(a, b)
            stat_name = 'ttest_rel'
        except Exception:
            pval = np.nan
            stat_name = None
            
    d = np.nan
    if n > 1:
        diff = a - b
        sd_diff = np.nanstd(diff, ddof=1)
        if sd_diff != 0:
            d = np.nanmean(diff) / sd_diff
    return {
        'n': n, 'mean_a': mean_a, 'mean_b': mean_b, 
        'pval': pval, 'stat_name': stat_name, 'cohens_d': d
    }

def mean_ci(a, alpha=0.05):
    a = np.asarray(a)
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0:
        return (np.nan, (np.nan, np.nan))
    mu = float(np.mean(a))
    if sem is not None and t is not None and n > 1:
        se = sem(a)
        crit = t.ppf(1 - alpha / 2.0, n - 1)
        return (mu, (mu - crit * se, mu + crit * se))
    else:
        se = np.std(a, ddof=1) / math.sqrt(n) if n > 1 else 0.0
        return (mu, (mu - 1.96 * se, mu + 1.96 * se))

# ---------- Plotting ----------
sns.set(style='whitegrid')
plt.rcParams.update({'font.size': 11})

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True)
axes_flat = axes.flatten()

for i, ax in enumerate(axes_flat):
    
    if i >= len(facets):
        ax.axis('off')
        continue
        
    facet = facets[i]
    sub = df[df[facet_col] == facet]
    pivot = sub.pivot(index='cell_id', columns='method', values='MTSCF').dropna()
    
    a_vals = pivot['MoE_Harmony'].values if 'MoE_Harmony' in pivot.columns else np.array([])
    b_vals = pivot['Harmony'].values if 'Harmony' in pivot.columns else np.array([])

    center_x = 0.0

    # 1. Draw Violins
    parts_a = ax.violinplot(a_vals, positions=[center_x], widths=VIOLIN_WIDTH,
                             showextrema=False, showmeans=False)
    parts_b = ax.violinplot(b_vals, positions=[center_x], widths=VIOLIN_WIDTH,
                             showextrema=False, showmeans=False)

    # Style MoE (Left Half)
    for pc in parts_a['bodies']:
        pc.set_facecolor(COLOR_MOE)
        pc.set_edgecolor('black')
        pc.set_alpha(VIOLIN_ALPHA) 
        path = pc.get_paths()[0]
        vertices = path.vertices
        cx = np.mean(vertices[:, 0])
        vertices[vertices[:, 0] > cx, 0] = cx
        path.vertices = vertices

    # Style Harmony (Right Half)
    for pc in parts_b['bodies']:
        pc.set_facecolor(COLOR_H)
        pc.set_edgecolor('black')
        pc.set_alpha(VIOLIN_ALPHA)
        path = pc.get_paths()[0]
        vertices = path.vertices
        cx = np.mean(vertices[:, 0])
        vertices[vertices[:, 0] < cx, 0] = cx
        path.vertices = vertices

    # 2. Draw Boxplots
    offset = 0.12
    if len(a_vals) > 0:
        bp_a = ax.boxplot(a_vals, positions=[center_x - offset], widths=BOX_WIDTH,
                          patch_artist=True, showfliers=False, manage_ticks=False)
        for box in bp_a['boxes']:
            box.set(facecolor=COLOR_MOE, alpha=VIOLIN_ALPHA, edgecolor='black')
        for item in ['medians', 'whiskers']:
            for line in bp_a.get(item, []): line.set(color='black', linewidth=0.9)

    if len(b_vals) > 0:
        bp_b = ax.boxplot(b_vals, positions=[center_x + offset], widths=BOX_WIDTH,
                          patch_artist=True, showfliers=False, manage_ticks=False)
        for box in bp_b['boxes']:
            box.set(facecolor=COLOR_H, alpha=VIOLIN_ALPHA, edgecolor='black')
        for item in ['medians', 'whiskers']:
            for line in bp_b.get(item, []): line.set(color='black', linewidth=0.9)

    # --- Points REMOVED here ---

    # 3. Mean + CI Markers
    if len(a_vals) > 0:
        mu_a, (lo_a, hi_a) = mean_ci(a_vals)
        ax.plot([center_x - offset], [mu_a], marker='D', color='white', markeredgecolor='black', markersize=6)
        ax.vlines(center_x - offset, lo_a, hi_a, color='black', linewidth=1.2)
    if len(b_vals) > 0:
        mu_b, (lo_b, hi_b) = mean_ci(b_vals)
        ax.plot([center_x + offset], [mu_b], marker='D', color='white', markeredgecolor='black', markersize=6)
        ax.vlines(center_x + offset, lo_b, hi_b, color='black', linewidth=1.2)

    # 4. Statistics Annotation
    stats = paired_statistics(a_vals, b_vals)
    pval = stats['pval']
    cohens_d = stats['cohens_d']
    stat_name = stats['stat_name'] if stats['stat_name'] else "test n/a"

    if np.isfinite(pval):
        p_txt = 'p < 1e-4' if pval < 1e-4 else f'p = {pval:.3g}'
    else:
        p_txt = 'p = n/a'
    d_txt = f"d = {cohens_d:.2f}" if cohens_d is not None and not np.isnan(cohens_d) else "d = n/a"

    y_max = max(np.nanmax(a_vals) if len(a_vals) else 0.0, np.nanmax(b_vals) if len(b_vals) else 0.0)
    y_lim_top = max(1.0, y_max + 0.12)
    ax.set_ylim(-0.02, y_lim_top)
    
    ann_text = f"{p_txt}\n{d_txt}\nn = {stats['n']}"
    ax.text(0.53, 1.05, ann_text, ha='left', va='top', 
            fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5, edgecolor='black'))

    # 5. X-Axis Label: Cluster 0, Cluster 1 ...
    ax.set_xticks([center_x])
    ax.set_xticklabels([f"Cluster {i}"], fontsize=14, fontweight='bold')
    ax.set_xlim(center_x - 0.8, center_x + 0.8)
    ax.set_ylim(-0.05, 1.1)

    ax.grid(True, which='major', 
        color='lightgray', 
        linestyle='--', 
        linewidth=1.5, 
        alpha=1)

# Global labels
fig.text(0.03, 0.5, 'MTSCF', va='center', rotation='vertical', fontsize=18, fontweight='bold')

# Legend
legend_elements = [
    Line2D([0], [0], marker='s', color='w', label='MoE-Harmony', 
           markerfacecolor=COLOR_MOE, markersize=10, alpha=VIOLIN_ALPHA),
    Line2D([0], [0], marker='s', color='w', label='Harmony', 
           markerfacecolor=COLOR_H, markersize=10, alpha=VIOLIN_ALPHA)
]
fig.legend(handles=legend_elements, 
           loc='lower center',  
           bbox_to_anchor=(0.5, -0.02),  
           frameon=True,
           columnspacing=2,
           handlelength=1.5,
           fontsize=14,
           ncol=2)  

plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.95])
plt.savefig(OUT_PATH, dpi=FIG_DPI, bbox_inches='tight')
print(f"Saved figure to: {OUT_PATH}")
plt.close(fig)