"""
Sen's Slope styled plots matching the reference image:
  Fig 1 – Horizontal bar chart: streamflow Sen's slope per gage
  Fig 2 – Per-gage WTE Sen's slope boxplot (IQR box, 5-95th whiskers, SF inset)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pymannkendall as mk
from pathlib import Path

BASE    = Path(__file__).parent.parent
OUT_DIR = BASE / "results/figures/senslope_styled"
OUT_CSV = BASE / "results/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_OBS   = 20
MIN_YEARS = 3

GAGE_SHORT = {
    '10126000': 'Bear River\nnr Corinne',
    '10141000': 'Weber River\nnr Plain City',
    '10142000': 'Farmington Cr\nnr Farmington',
    '10143500': 'Centerville Cr\nnr Centerville',
    '10152000': 'Spanish Fork\nnr Lake Shore',
    '10153100': 'Hobble Creek\n@ Springville',
    '10163000': 'Provo River\n@ Provo',
    '10168000': 'Little Cottonwood Cr',
    '10168500': 'Big Cottonwood Cr\nnr SLC',
    '10172700': 'Vernon Cr\nnr Vernon',
    '10172860': 'Warm Cr\nnr Gandy',
    '10172952': 'Dunn Cr\nnr Park Valley',
}

# ── Load data ──────────────────────────────────────────────────────────────────
wte_all = pd.read_csv(OUT_CSV / 'mk_well_wte.csv')
wte_all['gage_id'] = wte_all['gage_id'].astype(str)

sf_mk = pd.read_csv(OUT_CSV / 'mk_streamflow_bfd1.csv')
sf_mk['gage_id'] = sf_mk['gage_id'].astype(str)
sf_mk = sf_mk.set_index('gage_id')

# Filter WTE: n_obs >= MIN_OBS and year_span >= MIN_YEARS
wte = wte_all[(wte_all['n_obs'] >= MIN_OBS) & (wte_all['year_span'] >= MIN_YEARS)].copy()
print(f"WTE wells after filter (n≥{MIN_OBS}, ya≥{MIN_YEARS}): {len(wte)} / {len(wte_all)}")
print(f"  Per gage:\n{wte.groupby('gage_id').size().to_string()}")

# ── Save filtered WTE table ────────────────────────────────────────────────────
wte_out = wte[['well_id','gage_id','gage_name','n_obs','year_span',
               'date_start','date_end','trend','p_value',
               'sen_slope_yr','tau']].copy()
wte_out.columns = ['well_id','gage_id','gage_name','n_obs','year_span',
                   'date_start','date_end','trend','p_value',
                   'sen_slope_ft_yr','tau']
wte_out.to_csv(OUT_CSV / 'wte_senslope_filtered.csv', index=False)
print(f"\nSaved: wte_senslope_filtered.csv ({len(wte_out)} rows)")

# ── Fig 1: Streamflow Sen's slope horizontal bar chart ─────────────────────────
sf_plot = sf_mk.reset_index().sort_values('sen_slope_cfs_yr')
sf_plot['label'] = sf_plot.apply(
    lambda r: f"{r['gage_id']} {GAGE_SHORT.get(r['gage_id'],'').replace(chr(10),' ')} "
              + ('*' if r['p_value'] < 0.05 else ''),
    axis=1
)
colors = ['#D62728' if v < 0 else '#1F77B4' for v in sf_plot['sen_slope_cfs_yr']]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(sf_plot)), sf_plot['sen_slope_cfs_yr'],
               color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_yticks(range(len(sf_plot)))
ax.set_yticklabels(sf_plot['label'], fontsize=9)
ax.set_xlabel("Sen's slope (cfs/yr)", fontsize=11)
ax.set_title("Streamflow Trend — Sen's Slope per Gage\n"
             "(* p<0.05, red=declining, blue=increasing)", fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_axisbelow(True)

red_patch  = mpatches.Patch(color='#D62728', label='Declining')
blue_patch = mpatches.Patch(color='#1F77B4', label='Increasing')
ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc='lower right')

plt.tight_layout()
fig.savefig(OUT_DIR / 'fig1_streamflow_senslope_bar.png', dpi=160, bbox_inches='tight')
plt.close()
print("\nFig 1 saved: fig1_streamflow_senslope_bar.png")

# ── Fig 2: WTE Sen's slope boxplot per gage (styled) ──────────────────────────
gages_with_wte = sorted(wte['gage_id'].unique())
n_gages = len(gages_with_wte)
ncols = 3
nrows = int(np.ceil(n_gages / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
axes = np.array(axes).flatten()

for ax, gage_id in zip(axes, gages_with_wte):
    sub = wte[wte['gage_id'] == gage_id]['sen_slope_yr'].dropna()
    n_wells = len(sub)

    # Box: IQR; whiskers: 5-95th pct; no outliers
    q5, q25, q50, q75, q95 = np.percentile(sub, [5, 25, 50, 75, 95])
    clipped = sub.clip(lower=q5, upper=q95)

    bp = ax.boxplot(
        clipped,
        positions=[0],
        widths=0.5,
        patch_artist=True,
        whis=[5, 95],
        showfliers=False,
        medianprops=dict(color='#E8750A', linewidth=2.5),
        boxprops=dict(facecolor='#5B7FB5', alpha=0.8, linewidth=1.2),
        whiskerprops=dict(linewidth=1.2, linestyle='-'),
        capprops=dict(linewidth=1.2),
    )

    ax.axhline(0, color='#D62728', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels([f'n≥{MIN_OBS}, ya≥{MIN_YEARS}'], fontsize=9)
    ax.set_ylabel("WTE Sen's slope (ft/yr)", fontsize=9)
    ax.set_title(GAGE_SHORT.get(gage_id, gage_id).replace('\n', ' '),
                 fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # n_wells annotation
    ax.text(0.05, 0.97, f'n={n_wells}', transform=ax.transAxes,
            fontsize=9, va='top', ha='left', color='#333333')

    # ── Inset: streamflow Sen's slope ──────────────────────────────────────
    if gage_id in sf_mk.index:
        sf_row = sf_mk.loc[gage_id]
        slope  = sf_row['sen_slope_cfs_yr']
        pval   = sf_row['p_value']
        n_yr   = int(sf_row['n_years'])
        pct    = sf_row['pct_per_yr']
        sig    = '*' if pval < 0.05 else ''
        color  = '#D62728' if slope < 0 else '#1F77B4'
        txt = (f"Q (BFD=1) Sen's slope:\n"
               f"{slope:.3f} cfs/yr{sig}\n"
               f"({pct:+.2f}%/yr)\n"
               f"p = {pval:.3f}{sig}  (n={n_yr} yrs)")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                fontsize=7.5, va='top', ha='right',
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.85))

# Hide unused axes
for ax in axes[n_gages:]:
    ax.set_visible(False)

fig.suptitle(
    f"Per-well WTE Sen's Slope — Full Record\n"
    f"Filter: n≥{MIN_OBS}, ya≥{MIN_YEARS}  |  "
    f"box = IQR, whiskers = 5–95th pct, outliers hidden; "
    f"red dashed = 0",
    fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
fig.savefig(OUT_DIR / 'fig2_wte_senslope_boxplot.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig 2 saved: fig2_wte_senslope_boxplot.png")

# ── Summary table ──────────────────────────────────────────────────────────────
summary = (
    wte.groupby(['gage_id', 'gage_name'])['sen_slope_yr']
    .agg(
        n_wells='count',
        median_slope='median',
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75),
        pct_declining=lambda x: (x < 0).mean() * 100,
        pct_sig=lambda x: None,
    )
    .reset_index()
)
# pct significant
sig_counts = wte[wte['p_value'] < 0.05].groupby('gage_id').size().rename('n_sig')
summary = summary.merge(sig_counts, on='gage_id', how='left').fillna(0)
summary['pct_sig'] = (summary['n_sig'] / summary['n_wells'] * 100).round(1)
summary = summary.drop(columns=['pct_sig_x', 'pct_sig_y'], errors='ignore')
summary = summary.round(4)

# merge with streamflow
sf_merge = sf_mk.reset_index()[['gage_id','mean_q_cfs','sen_slope_cfs_yr','pct_per_yr','p_value']]
sf_merge.columns = ['gage_id','sf_mean_q_cfs','sf_sen_slope_cfs_yr','sf_pct_per_yr','sf_p_value']
summary = summary.merge(sf_merge, on='gage_id', how='left').round(4)
summary.to_csv(OUT_CSV / 'senslope_summary_table.csv', index=False)

print("\n=== Summary Table ===")
print(summary[['gage_id','n_wells','median_slope','pct_declining','pct_sig',
               'sf_sen_slope_cfs_yr','sf_pct_per_yr','sf_p_value']].to_string(index=False))
print(f"\nAll outputs → {OUT_DIR}")
print(f"Tables → {OUT_CSV}")
