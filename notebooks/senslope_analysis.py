"""
Sen's Slope Analysis:
1. Streamflow (bfd=1): Sen's slope per gage + Sen's slope / mean Q (%)
2. WTE: boxplot of Sen's slope per gage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pymannkendall as mk
from pathlib import Path

BASE    = Path(__file__).parent.parent
OUT_DIR = BASE / "results/figures/senslope"
OUT_CSV = BASE / "results/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAGE_SHORT = {
    '10126000': 'Bear River\nNr Corinne',
    '10141000': 'Weber River\nNr Plain City',
    '10142000': 'Farmington Cr\nNr Farmington',
    '10143500': 'Centerville Cr\nNr Centerville',
    '10152000': 'Spanish Fork\nNr Lake Shore',
    '10153100': 'Hobble Creek\n@ Springville',
    '10163000': 'Provo River\n@ Provo',
    '10168000': 'Little Cottonwood\nCr @ Jordan R',
    '10168500': 'Big Cottonwood\nCr Nr SLC',
    '10172700': 'Vernon Cr\nNr Vernon',
    '10172860': 'Warm Cr\nNr Gandy',
    '10172952': 'Dunn Cr\nNr Park Valley',
}

# ── 1. Streamflow Sen's Slope (bfd=1) ────────────────────────────────────────
print("Computing Sen's slope for bfd=1 streamflow …")
sf = pd.read_csv(BASE / "results/processed/streamflow_monthly_bfd.csv",
                 parse_dates=['date'])
sf['gage_id'] = sf['gage_id'].astype(str)
bfd1 = sf[sf['bfd'] == 1].copy()

sf_records = []
for gage_id, grp in bfd1.groupby('gage_id'):
    grp = grp.sort_values('date').dropna(subset=['q'])
    if len(grp) < 12:
        continue
    # annual mean to reduce seasonality before MK
    grp['year'] = grp['date'].dt.year
    annual = grp.groupby('year')['q'].mean()
    if len(annual) < 5:
        continue
    result = mk.original_test(annual.values)
    mean_q = grp['q'].mean()
    sen_yr  = result.slope          # cfs/year
    pct     = (sen_yr / mean_q) * 100 if mean_q > 0 else np.nan

    sf_records.append(dict(
        gage_id      = gage_id,
        gage_name    = GAGE_SHORT.get(gage_id, gage_id),
        n_months     = len(grp),
        n_years      = len(annual),
        mean_q_cfs   = round(mean_q, 3),
        sen_slope_cfs_yr = round(sen_yr, 4),
        pct_per_yr   = round(pct, 3),
        p_value      = round(result.p, 4),
        significant  = result.p < 0.05,
        trend        = result.trend,
    ))

sf_df = pd.DataFrame(sf_records).sort_values('gage_id')
sf_df.to_csv(OUT_CSV / 'mk_streamflow_bfd1.csv', index=False)

print("\nStreamflow (bfd=1) Sen's Slope:")
print(sf_df[['gage_id','gage_name','n_years','mean_q_cfs',
             'sen_slope_cfs_yr','pct_per_yr','p_value','trend']].to_string(index=False))

# ── Fig 1: Streamflow Sen's slope + % per yr (double-axis bar) ───────────────
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

gages  = sf_df['gage_id'].tolist()
labels = [GAGE_SHORT.get(g, g) for g in gages]
x      = np.arange(len(gages))
bar_w  = 0.35

colors_abs = ['#E15759' if v < 0 else '#59A14F' for v in sf_df['sen_slope_cfs_yr']]
colors_pct = ['#C0392B' if v < 0 else '#27AE60' for v in sf_df['pct_per_yr']]

bars1 = ax1.bar(x - bar_w/2, sf_df['sen_slope_cfs_yr'], bar_w,
                color=colors_abs, alpha=0.85, label="Sen's slope (cfs/yr)")
bars2 = ax2.bar(x + bar_w/2, sf_df['pct_per_yr'], bar_w,
                color=colors_pct, alpha=0.55, label="Sen's slope / mean Q (%/yr)")

# significance markers
for i, row in sf_df.reset_index(drop=True).iterrows():
    y1 = row['sen_slope_cfs_yr']
    y2 = row['pct_per_yr']
    if row['significant']:
        ax1.text(i - bar_w/2, y1 + (0.3 if y1 >= 0 else -0.6),
                 '*', ha='center', fontsize=14, color='black')
        ax2.text(i + bar_w/2, y2 + (0.05 if y2 >= 0 else -0.12),
                 '*', ha='center', fontsize=14, color='#555555')

ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Sen's Slope (cfs/yr)", fontsize=11)
ax2.set_ylabel("Sen's Slope / Mean Q (%/yr)", fontsize=11, color='#555555')
ax1.set_title("Mann-Kendall Sen's Slope — Baseflow-Dominated Streamflow (bfd=1)\n"
              "* = significant (p < 0.05)  |  Left: absolute (cfs/yr)  |  Right: % of mean Q",
              fontsize=12, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_axisbelow(True)
plt.tight_layout()
fig.savefig(OUT_DIR / 'senslope_streamflow_bfd1.png', dpi=160, bbox_inches='tight')
plt.close()
print("\n  Fig 1 saved: senslope_streamflow_bfd1.png")

# ── 2. WTE Sen's Slope Boxplot ────────────────────────────────────────────────
print("\nBuilding WTE Sen's slope boxplot …")
wte = pd.read_csv(OUT_CSV / 'mk_well_wte.csv')
wte['gage_id'] = wte['gage_id'].astype(str)

# sen_slope_yr is ft/year (annual Sen's slope on monthly data * 12)
# sen_slope is the raw monthly slope → convert to ft/yr
if 'sen_slope_yr' in wte.columns:
    wte['slope_ft_yr'] = wte['sen_slope_yr']
else:
    wte['slope_ft_yr'] = wte['sen_slope'] * 12

wte_gages = sorted(wte['gage_id'].unique())
wte_labels = [GAGE_SHORT.get(g, g) for g in wte_gages]

fig, ax = plt.subplots(figsize=(13, 6))

bp_data = [wte[wte['gage_id'] == g]['slope_ft_yr'].dropna().values for g in wte_gages]
COLORS  = plt.cm.tab10(np.linspace(0, 0.9, len(wte_gages)))

bp = ax.boxplot(bp_data, positions=np.arange(len(wte_gages)),
                patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))

for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# add median labels + n
for i, (vals, color) in enumerate(zip(bp_data, COLORS)):
    if len(vals) == 0:
        continue
    med = np.median(vals)
    n   = len(vals)
    ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] < -0.5 else med,
            '', ha='center')  # placeholder to fix ylim after draw
    ax.text(i, np.percentile(vals, 75) + 0.02,
            f'n={n}\nmed={med:.3f}', ha='center', va='bottom',
            fontsize=7.5, color='black')

ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.6)
ax.set_xticks(np.arange(len(wte_gages)))
ax.set_xticklabels(wte_labels, fontsize=9)
ax.set_ylabel("Sen's Slope (ft/yr)", fontsize=11)
ax.set_title("Distribution of Well WTE Sen's Slope by Gage\n"
             "(negative = declining groundwater level)",
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(OUT_DIR / 'senslope_wte_boxplot.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig 2 saved: senslope_wte_boxplot.png")

# ── Fig 3: WTE slope significant only, colour by direction ───────────────────
sig = wte[wte['p_value'] < 0.05].copy()
fig, ax = plt.subplots(figsize=(13, 6))

bp_sig = [sig[sig['gage_id'] == g]['slope_ft_yr'].dropna().values for g in wte_gages]

bp2 = ax.boxplot(bp_sig, positions=np.arange(len(wte_gages)),
                 patch_artist=True, notch=False,
                 medianprops=dict(color='black', linewidth=2),
                 whiskerprops=dict(linewidth=1.2),
                 capprops=dict(linewidth=1.2),
                 flierprops=dict(marker='o', markersize=3, alpha=0.4))

for patch, color in zip(bp2['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

for i, (vals, g) in enumerate(zip(bp_sig, wte_gages)):
    if len(vals) == 0:
        continue
    n_dec = (vals < 0).sum()
    n_inc = (vals >= 0).sum()
    med   = np.median(vals)
    top   = np.percentile(vals, 75)
    ax.text(i, top + 0.02,
            f'n={len(vals)}\n↓{n_dec} ↑{n_inc}\nmed={med:.3f}',
            ha='center', va='bottom', fontsize=7.5)

ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.6)
ax.set_xticks(np.arange(len(wte_gages)))
ax.set_xticklabels(wte_labels, fontsize=9)
ax.set_ylabel("Sen's Slope (ft/yr)", fontsize=11)
ax.set_title("Well WTE Sen's Slope — Significant Only (p < 0.05)\n"
             "↓ = declining  ↑ = rising  |  med = median slope",
             fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(OUT_DIR / 'senslope_wte_boxplot_sig.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig 3 saved: senslope_wte_boxplot_sig.png")

print(f"\nAll outputs → {OUT_DIR}")
