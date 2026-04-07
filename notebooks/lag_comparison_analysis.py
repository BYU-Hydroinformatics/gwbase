"""
Lag comparison analysis: no-lag vs 3mo vs 6mo vs 1yr
For each well-gage pair and each gage, compare R², slope, MI across lag periods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import mutual_info_score
from pathlib import Path

BASE    = Path(__file__).parent.parent
FEAT    = BASE / "results/features"
OUT_DIR = BASE / "results/figures/lag_comparison"
OUT_CSV = BASE / "results/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAGS = {
    'No Lag':  FEAT / 'data_with_deltas.csv',
    '3 Month': FEAT / 'data_lag_3mo.csv',
    '6 Month': FEAT / 'data_lag_6mo.csv',
    '1 Year':  FEAT / 'data_lag_1yr.csv',
}
LAG_ORDER  = list(LAGS.keys())
LAG_COLORS = ['#555555', '#4E79A7', '#F28E2B', '#E15759']

GAGE_NAME_MAP = {
    '10126000': 'Bear River\nNr Corinne',
    '10141000': 'Weber River\nNr Plain City',
    '10152000': 'Spanish Fork\nNr Lake Shore',
    '10153100': 'Hobble Creek\n@ Springville',
    '10163000': 'Provo River\n@ Provo',
    '10168000': 'Little Cottonwood\nCr @ Jordan R',
}

# ── helpers ──────────────────────────────────────────────────────────────────
def _mi(x, y, n_bins=20):
    """Mutual information via histogram discretisation."""
    x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
    y_bins = np.linspace(y.min(), y.max(), n_bins + 1)
    xd = np.digitize(x, x_bins) - 1
    yd = np.digitize(y, y_bins) - 1
    xd = np.clip(xd, 0, n_bins - 1)
    yd = np.clip(yd, 0, n_bins - 1)
    return mutual_info_score(xd, yd)

def _reg(x, y):
    if len(x) < 3 or x.std() == 0:
        return dict(slope=np.nan, r2=np.nan, p=np.nan, n=len(x))
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return dict(slope=slope, r2=r**2, p=p, n=len(x))

# ── compute stats per gage per lag ───────────────────────────────────────────
print("Computing regression and MI for each lag …")
gage_records = []
pair_records = []

for lag_label, path in LAGS.items():
    df = pd.read_csv(path)
    df['gage_id'] = df['gage_id'].astype(str)
    df = df.dropna(subset=['delta_wte', 'delta_q'])
    # Remove outlier for gage 10163000 (|delta_wte| > 1400 ft)
    df = df[~((df['gage_id'] == '10163000') & (df['delta_wte'].abs() > 1400))]

    for gage_id, g in df.groupby('gage_id'):
        gage_name = GAGE_NAME_MAP.get(gage_id, gage_id)
        x = g['delta_wte'].values
        y = g['delta_q'].values
        reg = _reg(x, y)
        mi  = _mi(x, y)
        gage_records.append(dict(
            lag=lag_label, gage_id=gage_id, gage_name=gage_name,
            n_obs=reg['n'], n_wells=g['well_id'].nunique(),
            **{k: reg[k] for k in ('slope','r2','p')},
            mi=mi
        ))

        # per-pair stats
        for well_id, w in g.groupby('well_id'):
            wx, wy = w['delta_wte'].values, w['delta_q'].values
            wreg = _reg(wx, wy)
            wmi  = _mi(wx, wy) if len(wx) >= 5 else np.nan
            pair_records.append(dict(
                lag=lag_label, gage_id=gage_id, gage_name=gage_name,
                well_id=well_id, **{f'{k}': wreg[k] for k in ('slope','r2','p')},
                mi=wmi
            ))

gage_df = pd.DataFrame(gage_records)
pair_df = pd.DataFrame(pair_records)

# save
gage_df.to_csv(OUT_CSV / 'lag_comparison_by_gage.csv', index=False)
pair_df.to_csv(OUT_CSV / 'lag_comparison_by_pair.csv', index=False)
print(f"  Gage-level rows: {len(gage_df)}  |  Pair-level rows: {len(pair_df)}")

# ── Figure 1: R² and MI by gage across lags (grouped bar) ────────────────────
gages = sorted(gage_df['gage_id'].unique())
n_gages = len(gages)
x_pos = np.arange(n_gages)
bar_w = 0.18

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

for i, lag in enumerate(LAG_ORDER):
    sub = gage_df[gage_df['lag'] == lag].set_index('gage_id')
    r2_vals = [sub.loc[g, 'r2'] if g in sub.index else np.nan for g in gages]
    mi_vals = [sub.loc[g, 'mi'] if g in sub.index else np.nan for g in gages]
    offset = (i - 1.5) * bar_w

    bars_r2 = axes[0].bar(x_pos + offset, r2_vals, bar_w,
                           color=LAG_COLORS[i], label=lag, alpha=0.85)
    bars_mi = axes[1].bar(x_pos + offset, mi_vals, bar_w,
                           color=LAG_COLORS[i], label=lag, alpha=0.85)

    # value labels on bars
    for bar, v in zip(bars_r2, r2_vals):
        if not np.isnan(v):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                         f'{v:.4f}', ha='center', va='bottom', fontsize=6.5, rotation=90)
    for bar, v in zip(bars_mi, mi_vals):
        if not np.isnan(v):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                         f'{v:.3f}', ha='center', va='bottom', fontsize=6.5, rotation=90)

tick_labels = [GAGE_NAME_MAP.get(g, g) for g in gages]
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(tick_labels, fontsize=9)
axes[0].set_ylabel('R²', fontsize=11)
axes[1].set_ylabel('Mutual Information', fontsize=11)
axes[0].set_title('ΔQ vs ΔWTE  —  R² by Gage and Lag Period', fontsize=13, fontweight='bold')
axes[1].set_title('ΔQ vs ΔWTE  —  Mutual Information by Gage and Lag Period', fontsize=13, fontweight='bold')
for ax in axes:
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.35)
    ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(OUT_DIR / 'lag_r2_mi_by_gage.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig 1 saved: lag_r2_mi_by_gage.png")

# ── Figure 2: pair-level R² distribution across lags (violin / box) ──────────
fig, axes = plt.subplots(1, n_gages, figsize=(14, 5), sharey=False)
if n_gages == 1:
    axes = [axes]

for ax, gage_id in zip(axes, gages):
    gage_name = GAGE_NAME_MAP.get(gage_id, gage_id)
    data_by_lag = []
    for lag in LAG_ORDER:
        sub = pair_df[(pair_df['gage_id'] == gage_id) & (pair_df['lag'] == lag)]['r2'].dropna()
        data_by_lag.append(sub.values)

    parts = ax.violinplot(data_by_lag, positions=range(len(LAG_ORDER)),
                          showmedians=True, showextrema=False)
    for pc, color in zip(parts['bodies'], LAG_COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    # overlay median text
    for j, vals in enumerate(data_by_lag):
        if len(vals) > 0:
            med = np.median(vals)
            ax.text(j, med + 0.003, f'{med:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(len(LAG_ORDER)))
    ax.set_xticklabels(['No\nLag', '3mo', '6mo', '1yr'], fontsize=8)
    ax.set_title(gage_name, fontsize=9, fontweight='bold')
    ax.set_xlabel('Lag', fontsize=8)
    ax.grid(axis='y', alpha=0.35)
    ax.set_axisbelow(True)

axes[0].set_ylabel('R² (per well-gage pair)', fontsize=10)
fig.suptitle('Distribution of R² across Well-Gage Pairs by Lag Period', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(OUT_DIR / 'lag_r2_violin_by_gage.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig 2 saved: lag_r2_violin_by_gage.png")

# ── Figure 3: delta R² heatmap (lag - no_lag) ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, max(4, n_gages * 0.55 + 1.5)))

for ax, lag in zip(axes, LAG_ORDER[1:]):   # skip no-lag itself
    no_lag_sub = gage_df[gage_df['lag'] == 'No Lag'].set_index('gage_id')
    lag_sub    = gage_df[gage_df['lag'] == lag].set_index('gage_id')

    delta_r2 = []
    delta_mi = []
    labels   = []
    for g in gages:
        if g in no_lag_sub.index and g in lag_sub.index:
            delta_r2.append(lag_sub.loc[g, 'r2'] - no_lag_sub.loc[g, 'r2'])
            delta_mi.append(lag_sub.loc[g, 'mi'] - no_lag_sub.loc[g, 'mi'])
            labels.append(GAGE_NAME_MAP.get(g, g))

    y = np.arange(len(labels))
    colors_r2 = ['#E15759' if v < 0 else '#59A14F' for v in delta_r2]
    colors_mi = ['#E15759' if v < 0 else '#4E79A7' for v in delta_mi]

    ax.barh(y - 0.18, delta_r2, 0.35, color=colors_r2, alpha=0.85, label='ΔR²')
    ax.barh(y + 0.18, delta_mi, 0.35, color=colors_mi, alpha=0.65, label='ΔMI')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f'{lag} vs No Lag\n(positive = lag better)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Δ (lag − no lag)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.35)
    ax.set_axisbelow(True)

# Unify x-axis scale across all three subplots
all_xlims = [ax.get_xlim() for ax in axes]
x_min = min(lim[0] for lim in all_xlims)
x_max = max(lim[1] for lim in all_xlims)
for ax in axes:
    ax.set_xlim(x_min, x_max)

fig.suptitle('Change in R² and MI with Lag (Gage Level)', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / 'lag_delta_r2_mi.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig 3 saved: lag_delta_r2_mi.png")

# ── Figure 4: pair-level % lag better summary ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for metric, ax in zip(['r2', 'mi'], axes):
    pct_better = []
    lag_labels_plot = []
    gage_labels_plot = []

    for lag in LAG_ORDER[1:]:
        for g in gages:
            no_lag_vals = pair_df[(pair_df['gage_id']==g)&(pair_df['lag']=='No Lag')]\
                          .set_index('well_id')[metric]
            lag_vals    = pair_df[(pair_df['gage_id']==g)&(pair_df['lag']==lag)]\
                          .set_index('well_id')[metric]
            common = no_lag_vals.index.intersection(lag_vals.index)
            if len(common) == 0:
                continue
            pct = ((lag_vals.loc[common] > no_lag_vals.loc[common]).sum() / len(common)) * 100
            pct_better.append(pct)
            lag_labels_plot.append(lag)
            gage_labels_plot.append(GAGE_NAME_MAP.get(g, g))

    summary = pd.DataFrame({'lag': lag_labels_plot, 'gage': gage_labels_plot, 'pct': pct_better})
    pivot = summary.pivot(index='gage', columns='lag', values='pct')[LAG_ORDER[1:]]

    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(len(LAG_ORDER[1:])))
    ax.set_xticklabels(LAG_ORDER[1:], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(LAG_ORDER[1:])):
            v = pivot.values[i, j]
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=9,
                    color='black' if 30 < v < 70 else 'white')
    ax.set_title(f'% of pairs where lag > no-lag\n(metric: {metric.upper()})',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label='% better')

fig.suptitle('Lag Effectiveness by Gage and Metric', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / 'lag_pct_better_heatmap.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig 4 saved: lag_pct_better_heatmap.png")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("LAG COMPARISON SUMMARY (Gage Level)")
print("="*70)
pivot_r2 = gage_df.pivot_table(index=['gage_id','gage_name'], columns='lag',
                                values='r2')[LAG_ORDER].round(4)
pivot_mi = gage_df.pivot_table(index=['gage_id','gage_name'], columns='lag',
                                values='mi')[LAG_ORDER].round(4)
print("\nR²:")
print(pivot_r2.to_string())
print("\nMutual Information:")
print(pivot_mi.to_string())
print(f"\nAll figures saved to: {OUT_DIR}")
print(f"CSVs saved to: {OUT_CSV}")
