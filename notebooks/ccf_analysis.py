"""
Cross-Correlation Function (CCF) Analysis
ΔWTE (groundwater) → ΔQ (streamflow), monthly paired data.

For each well-gage pair:
  CCF(lag) = corr(ΔQ[t], ΔWTE[t - lag])   lag = 0 … MAX_LAG months
  Positive lag: ΔWTE leads ΔQ (groundwater change precedes streamflow change)

Outputs
-------
- lag_ccf_pairs.csv          : per-pair CCF peak lag & peak correlation
- lag_ccf_gage_summary.csv   : per-gage summary statistics
- Fig 1: mean CCF curves per gage (± 95 % CI across wells)
- Fig 2: distribution of peak lags per gage (violin)
- Fig 3: heatmap — mean CCF(lag) per gage
- Fig 4: scatter — peak CCF vs n_obs, coloured by peak lag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
MAX_LAG   = 60   # months
MIN_OBS   = 36   # minimum pair observations required
BASE      = Path(__file__).parent.parent
OUT_FIG   = BASE / "results/figures/ccf_analysis"
OUT_CSV   = BASE / "results/features"
OUT_FIG.mkdir(parents=True, exist_ok=True)

GAGE_NAME = {
    '10126000': 'Bear River\nNr Corinne',
    '10141000': 'Weber River\nNr Plain City',
    '10152000': 'Spanish Fork\nNr Lake Shore',
    '10153100': 'Hobble Creek\n@ Springville',
    '10163000': 'Provo River\n@ Provo',
    '10168000': 'Little Cottonwood\nCr @ Jordan R',
}
GAGE_FULL = {
    '10126000': 'Bear River Nr Corinne',
    '10141000': 'Weber River Nr Plain City',
    '10152000': 'Spanish Fork Nr Lake Shore',
    '10153100': 'Hobble Creek @ Springville',
    '10163000': 'Provo River @ Provo',
    '10168000': 'Little Cottonwood Cr @ Jordan R',
}

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(BASE / "results/features/data_with_deltas.csv", parse_dates=['date'])
df['gage_id'] = df['gage_id'].astype(str)
df = df.dropna(subset=['delta_wte', 'delta_q'])
df = df.sort_values(['gage_id', 'well_id', 'date'])
print(f"  {len(df):,} records | {df['well_id'].nunique()} wells | "
      f"{df['gage_id'].nunique()} gages")

# ── CCF per pair ──────────────────────────────────────────────────────────────
def ccf_pair(x, y, max_lag):
    """
    CCF of y=ΔQ with lagged x=ΔWTE.
    Returns array of length max_lag+1 (lags 0..max_lag).
    Normalised to [-1, 1].
    """
    n = len(x)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    result = np.full(max_lag + 1, np.nan)
    for lag in range(max_lag + 1):
        if n - lag < 5:
            break
        result[lag] = np.corrcoef(x[:n - lag], y[lag:])[0, 1]
    return result

print("Computing CCF for each well-gage pair …")
pair_records = []
ccf_by_gage  = {g: [] for g in df['gage_id'].unique()}

for (gage_id, well_id), grp in df.groupby(['gage_id', 'well_id']):
    grp = grp.sort_values('date')
    n = len(grp)
    if n < MIN_OBS:
        continue

    x = grp['delta_wte'].values   # ΔWTE (predictor)
    y = grp['delta_q'].values     # ΔQ   (response)

    ccf_vals = ccf_pair(x, y, MAX_LAG)
    valid     = ~np.isnan(ccf_vals)
    if not valid.any():
        continue

    lags       = np.arange(MAX_LAG + 1)[valid]
    corrs      = ccf_vals[valid]
    peak_idx   = np.argmax(np.abs(corrs))
    peak_lag   = lags[peak_idx]
    peak_corr  = corrs[peak_idx]
    ci95       = 1.96 / np.sqrt(n)

    pair_records.append({
        'gage_id':    gage_id,
        'gage_name':  GAGE_FULL.get(gage_id, gage_id),
        'well_id':    well_id,
        'n_obs':      n,
        'peak_lag_months': peak_lag,
        'peak_corr':  peak_corr,
        'abs_peak':   abs(peak_corr),
        'ci95':       ci95,
        'sig_at_0':   abs(ccf_vals[0]) > ci95,
        'ccf_at_0':   ccf_vals[0],
        **{f'ccf_lag{lag}': ccf_vals[lag] if lag <= MAX_LAG else np.nan
           for lag in [0, 3, 6, 12, 24, 36]},
    })
    ccf_by_gage[gage_id].append(ccf_vals)

pairs_df = pd.DataFrame(pair_records)
pairs_df.to_csv(OUT_CSV / 'ccf_pairs.csv', index=False)
print(f"  {len(pairs_df)} pairs processed")

# ── gage-level summary ────────────────────────────────────────────────────────
gage_summary = (
    pairs_df.groupby(['gage_id', 'gage_name'])
    .agg(
        n_pairs          = ('well_id',          'count'),
        median_peak_lag  = ('peak_lag_months',  'median'),
        mean_peak_lag    = ('peak_lag_months',  'mean'),
        mean_abs_peak    = ('abs_peak',         'mean'),
        pct_sig_at_0     = ('sig_at_0',         lambda x: x.mean() * 100),
        mean_ccf_at_0    = ('ccf_at_0',         'mean'),
        mean_ccf_lag6    = ('ccf_lag6',          'mean'),
        mean_ccf_lag12   = ('ccf_lag12',         'mean'),
        mean_ccf_lag24   = ('ccf_lag24',         'mean'),
    )
    .reset_index()
    .round(4)
)
gage_summary.to_csv(OUT_CSV / 'ccf_gage_summary.csv', index=False)

print("\nGage-level CCF Summary:")
print(gage_summary[['gage_id','n_pairs','median_peak_lag','mean_abs_peak',
                     'pct_sig_at_0','mean_ccf_at_0']].to_string(index=False))

# ── Fig 1: Mean CCF curves per gage ──────────────────────────────────────────
gages   = sorted(pairs_df['gage_id'].unique())
n_gages = len(gages)
COLORS  = plt.cm.tab10(np.linspace(0, 0.9, n_gages))
lags_x  = np.arange(MAX_LAG + 1)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
axes = axes.flatten()

for ax, (gage_id, color) in zip(axes, zip(gages, COLORS)):
    curves = [c for c in ccf_by_gage[gage_id] if not np.all(np.isnan(c))]
    if not curves:
        ax.set_visible(False)
        continue

    mat  = np.vstack(curves)                       # (n_wells, max_lag+1)
    mean = np.nanmean(mat, axis=0)
    se   = np.nanstd(mat, axis=0) / np.sqrt((~np.isnan(mat)).sum(axis=0))
    n_curves = len(curves)
    ci95_bound = 1.96 / np.sqrt(pairs_df[pairs_df['gage_id']==gage_id]['n_obs'].median())

    ax.fill_between(lags_x, mean - 1.96*se, mean + 1.96*se,
                    alpha=0.2, color=color)
    ax.plot(lags_x, mean, color=color, linewidth=2, label='Mean CCF')

    # 95% significance threshold (median n_obs)
    ax.axhline( ci95_bound, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.axhline(-ci95_bound, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)

    # mark peak
    peak_lag  = int(np.argmax(np.abs(mean)))
    peak_val  = mean[peak_lag]
    ax.scatter([peak_lag], [peak_val], color='red', zorder=5, s=60)
    ax.annotate(f'peak={peak_lag}mo\n({peak_val:+.3f})',
                xy=(peak_lag, peak_val),
                xytext=(peak_lag + 3, peak_val + (0.02 if peak_val >= 0 else -0.04)),
                fontsize=8, color='red')

    ax.set_title(f"{GAGE_NAME.get(gage_id, gage_id)}\n(n={n_curves} wells)",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Lag (months)', fontsize=9)
    ax.set_ylabel('CCF(ΔQ, ΔWTE)', fontsize=9)
    ax.set_xlim(0, MAX_LAG)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

fig.suptitle('Cross-Correlation Function: ΔQ vs ΔWTE\n'
             '(positive lag → ΔWTE leads ΔQ)\n'
             'Shaded band = ±1 SE across wells  |  Dashed = 95% CI',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_FIG / 'ccf_curves_by_gage.png', dpi=160, bbox_inches='tight')
plt.close()
print("\n  Fig 1 saved: ccf_curves_by_gage.png")

# ── Fig 2: Distribution of peak lags per gage (violin) ───────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

positions = np.arange(n_gages)
for i, (gage_id, color) in enumerate(zip(gages, COLORS)):
    sub = pairs_df[pairs_df['gage_id'] == gage_id]['peak_lag_months'].values
    if len(sub) == 0:
        continue
    parts = ax.violinplot([sub], positions=[i], showmedians=True,
                          showextrema=True, widths=0.7)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.75)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    parts['cmins'].set_color(color)
    parts['cmaxes'].set_color(color)
    parts['cbars'].set_color(color)

    med = np.median(sub)
    ax.text(i, med + 1, f'{med:.0f}mo', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

ax.set_xticks(positions)
ax.set_xticklabels([GAGE_NAME.get(g, g) for g in gages], fontsize=9)
ax.set_ylabel('Peak Lag (months)', fontsize=11)
ax.set_title('Distribution of Optimal CCF Lag by Gage\n'
             '(lag at which |CCF| is maximised per well-gage pair)',
             fontsize=13, fontweight='bold')
ax.set_ylim(-2, MAX_LAG + 5)
ax.axhline(12, color='steelblue', linestyle=':', linewidth=1, alpha=0.7, label='12 mo')
ax.axhline(24, color='darkorange', linestyle=':', linewidth=1, alpha=0.7, label='24 mo')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.35)
ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(OUT_FIG / 'ccf_peak_lag_violin.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig 2 saved: ccf_peak_lag_violin.png")

# ── Fig 3: Heatmap — mean CCF(lag) per gage ──────────────────────────────────
heatmap_lags = list(range(0, MAX_LAG + 1, 3))    # every 3 months
heatmap_data = np.full((n_gages, len(heatmap_lags)), np.nan)

for i, gage_id in enumerate(gages):
    curves = [c for c in ccf_by_gage[gage_id] if not np.all(np.isnan(c))]
    if not curves:
        continue
    mat  = np.vstack(curves)
    mean = np.nanmean(mat, axis=0)
    for j, lag in enumerate(heatmap_lags):
        if lag < len(mean):
            heatmap_data[i, j] = mean[lag]

fig, ax = plt.subplots(figsize=(16, 4.5))
vmax = np.nanmax(np.abs(heatmap_data))
im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
               vmin=-vmax, vmax=vmax, interpolation='nearest')

ax.set_xticks(range(len(heatmap_lags)))
ax.set_xticklabels(heatmap_lags, fontsize=8)
ax.set_yticks(range(n_gages))
ax.set_yticklabels([GAGE_FULL.get(g, g) for g in gages], fontsize=9)
ax.set_xlabel('Lag (months)', fontsize=11)
ax.set_title('Mean CCF(ΔQ, ΔWTE) by Gage and Lag\n'
             '(red = positive correlation, blue = negative)',
             fontsize=13, fontweight='bold')

# annotate cells
for i in range(n_gages):
    for j in range(len(heatmap_lags)):
        v = heatmap_data[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(v) > vmax * 0.5 else 'black')

plt.colorbar(im, ax=ax, label='Mean CCF', shrink=0.8)
plt.tight_layout()
fig.savefig(OUT_FIG / 'ccf_heatmap.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig 3 saved: ccf_heatmap.png")

# ── Fig 4: peak CCF vs n_obs, coloured by peak lag ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# left: scatter peak|CCF| vs n_obs
sc = axes[0].scatter(
    pairs_df['n_obs'], pairs_df['abs_peak'],
    c=pairs_df['peak_lag_months'], cmap='viridis',
    s=50, alpha=0.7, edgecolors='none'
)
plt.colorbar(sc, ax=axes[0], label='Peak lag (months)')
axes[0].set_xlabel('Number of observations (months)', fontsize=11)
axes[0].set_ylabel('|Peak CCF|', fontsize=11)
axes[0].set_title('Peak CCF Magnitude vs Record Length\n(colour = optimal lag)',
                  fontsize=11, fontweight='bold')
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].grid(alpha=0.3)
for gid, color in zip(gages, COLORS):
    sub = pairs_df[pairs_df['gage_id'] == gid]
    axes[0].scatter([], [], color=color, label=GAGE_FULL.get(gid, gid), s=40)
axes[0].legend(fontsize=7, loc='upper right')

# right: CCF at key lags per gage (box)
key_lags = [0, 6, 12, 24]
lag_cols  = [f'ccf_lag{l}' for l in key_lags]
gage_labels_long = [GAGE_FULL.get(g, g) for g in gages]

x_pos = np.arange(len(key_lags))
bar_w = 0.13
for i, gage_id in enumerate(gages):
    sub = pairs_df[pairs_df['gage_id'] == gage_id]
    means = [sub[col].mean() for col in lag_cols]
    sems  = [sub[col].sem()  for col in lag_cols]
    offset = (i - len(gages)/2 + 0.5) * bar_w
    bars = axes[1].bar(x_pos + offset, means, bar_w,
                       color=COLORS[i], alpha=0.85,
                       label=GAGE_FULL.get(gage_id, gage_id))
    axes[1].errorbar(x_pos + offset, means, yerr=sems,
                     fmt='none', color='black', linewidth=1, capsize=2)

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([f'Lag {l} mo' for l in key_lags], fontsize=10)
axes[1].axhline(0, color='black', linewidth=0.7)
axes[1].set_ylabel('Mean CCF (across pairs)', fontsize=11)
axes[1].set_title('Mean CCF at Key Lags by Gage\n(error bars = ±1 SE)',
                  fontsize=11, fontweight='bold')
axes[1].legend(fontsize=7, loc='upper right')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_axisbelow(True)

plt.tight_layout()
fig.savefig(OUT_FIG / 'ccf_peak_vs_obs.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig 4 saved: ccf_peak_vs_obs.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("CCF ANALYSIS SUMMARY")
print("="*65)
cols = ['gage_id','n_pairs','median_peak_lag','mean_abs_peak',
        'pct_sig_at_0','mean_ccf_at_0','mean_ccf_lag6','mean_ccf_lag12']
print(gage_summary[cols].to_string(index=False))
print(f"\nAll figures → {OUT_FIG}")
print(f"CSVs        → {OUT_CSV}")
