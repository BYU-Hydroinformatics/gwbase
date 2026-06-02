"""
Filter Sensitivity Analysis
============================
Produces tables and figures comparing:
  1. Temporal quality filter (min_years × min_obs combinations)
  2. Elevation buffer (10 / 20 / 30 / 50 / 100 m)

Outputs written to results/analysis/filter_sensitivity/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'analysis', 'filter_sensitivity')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────

PROC = os.path.join(os.path.dirname(__file__), '..', 'results', 'processed')
RAW_TS = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',
                      'groundwater', 'GSLB_1900-2025_TS_with_aquifers.csv')

print("Loading data...")

# Wells in catchments (Step 2 output — sets the population before Step 4)
wic = pd.read_csv(os.path.join(PROC, 'wells_in_catchments.csv'))
watershed_well_ids = set(wic['well_id'].astype(str))

# Raw well time series, restricted to watershed wells
raw_ts = pd.read_csv(RAW_TS)
raw_ts.columns = [c.lower() for c in raw_ts.columns]
raw_ts = raw_ts.rename(columns={'well_id': 'well_id', 'date': 'date', 'wte': 'wte'})
raw_ts['well_id'] = raw_ts['well_id'].astype(str)
raw_ts = raw_ts[raw_ts['well_id'].isin(watershed_well_ids)].copy()
raw_ts['date'] = pd.to_datetime(raw_ts['date'])

# Compute per-well stats once (fast)
well_stats = (
    raw_ts.groupby('well_id')['date']
    .agg(n_obs='count',
         date_min='min',
         date_max='max')
    .reset_index()
)
well_stats['year_span'] = (well_stats['date_max'] - well_stats['date_min']).dt.days / 365.25
n_wells_total = len(well_stats)

# Interpolated monthly data + well-reach (for elevation filter)
monthly = pd.read_csv(os.path.join(PROC, 'well_pchip_monthly.csv'))
monthly['well_id'] = monthly['well_id'].astype(str)
monthly['date'] = pd.to_datetime(monthly['date'])

well_reach = pd.read_csv(os.path.join(PROC, 'well_reach_relationships.csv'))
well_reach = well_reach.rename(columns={
    'Well_ID': 'well_id',
    'Reach_Elevation': 'reach_elev_m'
})
well_reach['well_id'] = well_reach['well_id'].astype(str)
# keep only needed cols
well_reach = well_reach[['well_id', 'reach_elev_m']].drop_duplicates('well_id')

# Merge elevation info onto monthly
monthly_elev = monthly.merge(well_reach, on='well_id', how='inner')
monthly_elev['wte_m'] = monthly_elev['wte'] * 0.3048
monthly_elev['delta_elev'] = monthly_elev['reach_elev_m'] - monthly_elev['wte_m']

# Wells that pass Step 4 (our baseline 3yr+20obs) — for pairing context
baseline_wells = set(
    well_stats.loc[
        (well_stats['n_obs'] >= 20) & (well_stats['year_span'] >= 3), 'well_id'
    ]
)

print(f"  Watershed wells (pre-filter): {n_wells_total}")
print(f"  Baseline (3yr+20obs) wells:   {len(baseline_wells)}")

# ── 2. Temporal filter sensitivity ───────────────────────────────────────────

print("\nRunning temporal filter sensitivity...")

year_options = [1, 2, 3, 5, 7, 10]
obs_options  = [5, 10, 20, 30, 50]

rows = []
for min_yr in year_options:
    for min_obs in obs_options:
        mask = (well_stats['n_obs'] >= min_obs) & (well_stats['year_span'] >= min_yr)
        n_pass = mask.sum()
        rows.append({
            'min_years': min_yr,
            'min_obs':   min_obs,
            'n_wells':   n_pass,
            'pct_wells': round(n_pass / n_wells_total * 100, 1),
        })

temporal_df = pd.DataFrame(rows)
temporal_df.to_csv(os.path.join(OUT_DIR, 'temporal_filter_sensitivity.csv'), index=False)

# Pivot to matrix for display
pivot_n   = temporal_df.pivot(index='min_years', columns='min_obs', values='n_wells')
pivot_pct = temporal_df.pivot(index='min_years', columns='min_obs', values='pct_wells')
print("\nWells retained (count):")
print(pivot_n.to_string())
print("\nWells retained (%):")
print(pivot_pct.to_string())

# ── 3. Elevation buffer sensitivity ──────────────────────────────────────────

print("\nRunning elevation buffer sensitivity...")

buffer_vals = [10, 20, 30, 50, 100]
n_wells_pre_elev = monthly_elev['well_id'].nunique()
n_records_pre    = len(monthly_elev)

elev_rows = []
for buf in buffer_vals:
    kept = monthly_elev[monthly_elev['delta_elev'] <= buf]
    n_w = kept['well_id'].nunique()
    n_r = len(kept)
    elev_rows.append({
        'buffer_m':    buf,
        'n_wells':     n_w,
        'n_records':   n_r,
        'pct_wells':   round(n_w / n_wells_pre_elev * 100, 1),
        'pct_records': round(n_r / n_records_pre   * 100, 1),
    })

elev_df = pd.DataFrame(elev_rows)
elev_df.to_csv(os.path.join(OUT_DIR, 'elevation_filter_sensitivity.csv'), index=False)
print(elev_df.to_string(index=False))

# Delta-elevation distribution (for all baseline wells)
baseline_monthly = monthly_elev[monthly_elev['well_id'].isin(baseline_wells)]
delta_bins   = [-np.inf, -20, -10, -5, 0, 5, 10, 20, 30, 50, 75, 100, np.inf]
bin_labels   = ['< -20', '-20–-10', '-10–-5', '-5–0',
                 '0–5', '5–10', '10–20', '20–30',
                 '30–50', '50–75', '75–100', '≥ 100']
baseline_monthly = baseline_monthly.copy()
baseline_monthly['delta_bin'] = pd.cut(
    baseline_monthly['delta_elev'], bins=delta_bins, labels=bin_labels
)
delta_dist = (
    baseline_monthly.groupby('delta_bin', observed=True)
    .size()
    .reset_index(name='count')
)
delta_dist['pct'] = (delta_dist['count'] / len(baseline_monthly) * 100).round(2)
delta_dist.to_csv(os.path.join(OUT_DIR, 'delta_elevation_distribution.csv'), index=False)

# ── 4. Plots ──────────────────────────────────────────────────────────────────

COLORS = {
    5:  '#d62728',
    10: '#ff7f0e',
    20: '#2ca02c',   # chosen threshold
    30: '#1f77b4',
    50: '#9467bd',
}

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
})

# ── Fig 1: Line plot — temporal quality filter ────────────────────────────────

fig1, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

for obs in obs_options:
    sub = temporal_df[temporal_df['min_obs'] == obs].sort_values('min_years')
    ls = '-' if obs == 20 else '--'
    lw = 2.2 if obs == 20 else 1.3
    ax.plot(sub['min_years'], sub['pct_wells'],
            color=COLORS[obs], linestyle=ls, linewidth=lw,
            marker='o', markersize=5,
            label=f'≥{obs} obs')

ax.axvline(3, color='black', linestyle=':', linewidth=1.2, alpha=0.7)
ax.set_xlabel('Minimum record length (years)')
ax.set_ylabel('Wells retained (%)')
ax.set_title('Temporal quality filter sensitivity')
ax.set_xticks(year_options)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
# Place legend below the plot to avoid overlapping lines
ax.legend(
    title='Min. observations', fontsize=9, title_fontsize=9,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
    frameon=True,
)
ax.grid(True, alpha=0.3)

# Annotate chosen threshold
chosen_row = temporal_df[
    (temporal_df['min_years'] == 3) & (temporal_df['min_obs'] == 20)
].iloc[0]
ax.annotate(
    f"Chosen: 3 yr + 20 obs\n({int(chosen_row['n_wells'])} wells, {chosen_row['pct_wells']:.0f}%)",
    xy=(3, chosen_row['pct_wells']),
    xytext=(5.5, chosen_row['pct_wells'] + 4),
    fontsize=8.5,
    arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.9),
)

fig1.savefig(os.path.join(OUT_DIR, 'temporal_filter_sensitivity.png'),
             dpi=600, bbox_inches='tight')
plt.close(fig1)
print("\nSaved: temporal_filter_sensitivity.png")

# ── Fig 2: Bar chart — elevation buffer ──────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(6.5, 5), constrained_layout=True)

bar_colors = ['#aec7e8', '#aec7e8', '#2ca02c', '#aec7e8', '#aec7e8']
bars = ax2.bar(
    [str(b) + ' m' for b in buffer_vals],
    elev_df['pct_wells'],
    color=bar_colors,
    edgecolor='grey',
    linewidth=0.7,
    width=0.55,
)
ax2.set_xlabel('Elevation buffer below streambed')
ax2.set_ylabel('Wells retained (%)')
ax2.set_title('Elevation buffer filter sensitivity')
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax2.grid(True, axis='y', alpha=0.3)

for bar, row in zip(bars, elev_df.itertuples()):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f'{row.n_wells}',
        ha='center', va='bottom', fontsize=9,
    )

ax2.get_xticklabels()[2].set_fontweight('bold')
ax2.annotate(
    'Chosen (30 m)',
    xy=(2, elev_df.loc[elev_df['buffer_m'] == 30, 'pct_wells'].values[0]),
    xytext=(3.2, elev_df.loc[elev_df['buffer_m'] == 30, 'pct_wells'].values[0] - 10),
    fontsize=8.5,
    arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.9),
)

fig2.savefig(os.path.join(OUT_DIR, 'elevation_buffer_sensitivity.png'),
             dpi=600, bbox_inches='tight')
plt.close(fig2)
print("Saved: elevation_buffer_sensitivity.png")

# ── Fig 2: Delta-elevation distribution (cumulative bar, shown as waterfall) ──

fig2, ax3 = plt.subplots(figsize=(9, 4), constrained_layout=True)

colors_bar = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(delta_dist)))
bars2 = ax3.bar(
    range(len(delta_dist)),
    delta_dist['pct'],
    color=colors_bar,
    edgecolor='white',
    linewidth=0.5,
)
ax3.set_xticks(range(len(delta_dist)))
ax3.set_xticklabels(delta_dist['delta_bin'].astype(str), rotation=35, ha='right', fontsize=9)
ax3.set_ylabel('Monthly records (%)')
ax3.set_xlabel('Δ elevation: streambed minus WTE (m)  [positive = WTE below stream]')
ax3.set_title('Distribution of Δ elevation (baseline wells, post-interpolation)')
ax3.grid(True, axis='y', alpha=0.3)

# Mark cutoff at 30m
cutoff_idx = list(delta_dist['delta_bin'].astype(str)).index('20–30')
vline = ax3.axvline(cutoff_idx + 0.5, color='red', linestyle='--', linewidth=1.5)

# Cumulative line on secondary axis
ax3b = ax3.twinx()
cumsum = delta_dist['pct'].cumsum()
cum_line, = ax3b.plot(range(len(delta_dist)), cumsum, 'k--o', markersize=4,
                      linewidth=1.3)
ax3b.set_ylabel('Cumulative records (%)')
ax3b.set_ylim(0, 108)
ax3b.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))

# Single combined legend on ax3b, placed lower-right where cumulative line is ~90%+
# and bars are short — no overlap there
ax3b.legend(
    [cum_line, vline],
    ['Cumulative %', '30 m cutoff'],
    loc='upper left',
    fontsize=9,
    framealpha=0.9,
)

fig2.savefig(os.path.join(OUT_DIR, 'delta_elevation_distribution.png'),
             dpi=600, bbox_inches='tight')
plt.close(fig2)
print("Saved: delta_elevation_distribution.png")

# ── 5. Summary table printout ─────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY TABLES")
print("="*60)

print("\nTable 1 — Temporal filter: wells retained by min_years × min_obs")
print(f"  (Starting population: {n_wells_total} watershed wells)\n")
tbl = pivot_n.copy()
tbl.columns = [f'≥{c} obs' for c in tbl.columns]
tbl.index.name = 'min_years'
print(tbl.to_string())

print("\nTable 2 — Elevation buffer: wells and records retained")
print(f"  (Pre-filter: {n_wells_pre_elev} wells, {n_records_pre:,} monthly records)\n")
print(elev_df.to_string(index=False))

print(f"\nOutputs in: {os.path.abspath(OUT_DIR)}")
