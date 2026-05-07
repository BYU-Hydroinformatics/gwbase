#!/usr/bin/env python3
"""
Per-well MK + Sen's slope restricted to last 50 years (1976+).
Whisker plots (box plots) per basin for each filter scenario.

Output: report/comparison_pts_yr3/mk_well_last50yr_whisker.png
         report/comparison_pts_yr3/mk_well_last50yr_table.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymannkendall as mk

# ── Config ───────────────────────────────────────────────────────────────────

SCENARIOS = [
    ('filter_pt2_yr3',  'n≥2, y≥3'),
    ('filter_pt10_yr3', 'n≥10, y≥3'),
    ('filter_pt20_yr3', 'n≥20, y≥3'),
    ('filter_pt30_yr3', 'n≥30, y≥3'),
]
REPORT_BASE = 'report'
OUT_DIR = 'report/comparison_pts_yr3'
CUTOFF_50YR = pd.Timestamp('1976-01-01')
MIN_OBS = 10   # minimum observations after 1976 cutoff to include well

GAGE_ORDER = ['10126000', '10141000', '10168000']
GAGE_NAMES = {
    '10126000': 'Bear River nr Corinne',
    '10141000': 'Weber River nr Plain City',
    '10168000': 'Little Cottonwood Cr',
}
COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']


# ── MK helper ────────────────────────────────────────────────────────────────

def run_mk_well(grp: pd.DataFrame, wte_col='wte', date_col='date'):
    """Run MK + Sen's slope on a single well's time series. Returns None if insufficient."""
    grp = grp.dropna(subset=[wte_col]).sort_values(date_col)
    n = len(grp)
    if n < MIN_OBS:
        return None
    try:
        res = mk.original_test(grp[wte_col].values)
    except Exception:
        return None

    # Convert monthly slope → ft/yr via median interval
    intervals = grp[date_col].diff().dt.days.dropna()
    med_days = intervals.median()
    slope_yr = res.slope * (365.25 / med_days) if med_days > 0 else np.nan

    yr_span = (grp[date_col].max() - grp[date_col].min()).days / 365.25
    return {
        'n_obs': n,
        'year_span': round(yr_span, 2),
        'date_start': grp[date_col].min().date(),
        'date_end':   grp[date_col].max().date(),
        'trend':  res.trend,
        'h':      bool(res.h),
        'p_value': float(res.p),
        'tau':    float(res.Tau),
        'sen_slope_yr': slope_yr,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def compute_scenario(scenario_dir: str, label: str) -> pd.DataFrame:
    path = os.path.join(REPORT_BASE, scenario_dir, 'features', 'data_with_deltas.csv')
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=['date'])

    # Keep only ≥ 1976
    df = df[df['date'] >= CUTOFF_50YR].copy()

    # Retain gage metadata
    gage_map = df[['well_id', 'gage_id', 'gage_name']].drop_duplicates('well_id').set_index('well_id')

    rows = []
    for wid, grp in df.groupby('well_id'):
        result = run_mk_well(grp)
        if result is None:
            continue
        result['well_id'] = wid
        result['gage_id'] = str(gage_map.loc[wid, 'gage_id']) if wid in gage_map.index else np.nan
        result['gage_name'] = gage_map.loc[wid, 'gage_name'] if wid in gage_map.index else ''
        result['scenario'] = label
        rows.append(result)

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = []
    for scenario_dir, label in SCENARIOS:
        print(f"Processing {scenario_dir} ({label}) ...")
        res = compute_scenario(scenario_dir, label)
        print(f"  → {len(res)} wells with ≥{MIN_OBS} obs after 1976")
        all_results.append((label, res))

    # ── Save table ──────────────────────────────────────────────────────────
    combined = pd.concat([r for _, r in all_results], ignore_index=True)
    col_order = ['scenario', 'well_id', 'gage_id', 'gage_name',
                 'n_obs', 'year_span', 'date_start', 'date_end',
                 'trend', 'h', 'p_value', 'tau', 'sen_slope_yr']
    combined = combined[[c for c in col_order if c in combined.columns]]
    out_csv = os.path.join(OUT_DIR, 'mk_well_last50yr_table.csv')
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved table → {out_csv}")

    # ── Whisker plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax_idx, gid in enumerate(GAGE_ORDER):
        ax = axes[ax_idx]
        data_list, labels_list = [], []

        for label, res in all_results:
            if res.empty or 'gage_id' not in res.columns:
                continue
            grp = res[res['gage_id'] == gid]['sen_slope_yr'].dropna()
            if len(grp):
                data_list.append(grp.values)
                labels_list.append(label)

        if data_list:
            bp = ax.boxplot(data_list, tick_labels=labels_list,
                            patch_artist=True, notch=False, showfliers=False)
            for patch, color in zip(bp['boxes'], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Annotate n wells
            for i, (vals, lbl) in enumerate(zip(data_list, labels_list), start=1):
                ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else min(v.min() for v in data_list),
                        f'n={len(vals)}', ha='center', va='top', fontsize=7, color='#555')

        ax.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.7)
        ax.set_title(f'{GAGE_NAMES.get(gid, gid)}', fontsize=9)
        ax.set_ylabel("Sen's slope (ft/yr)")
        ax.tick_params(axis='x', rotation=15, labelsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        "Per-well Sen's Slope Distribution — WTE, last 50 years (1976+)\n"
        "(outliers hidden; red dashed = 0; min 10 obs per well)",
        fontsize=11
    )
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'mk_well_last50yr_whisker.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot  → {out_png}")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\nSummary per scenario / gage:")
    for label, res in all_results:
        if res.empty:
            continue
        print(f"\n  {label}")
        for gid in GAGE_ORDER:
            g = res[res['gage_id'] == gid]
            if g.empty:
                continue
            sig = (g['p_value'] < 0.05).sum()
            med = g['sen_slope_yr'].median()
            print(f"    {GAGE_NAMES.get(gid,gid):35s}  n={len(g):3d}  "
                  f"sig={sig:3d} ({sig/len(g)*100:.0f}%)  "
                  f"median slope={med:+.4f} ft/yr")


if __name__ == '__main__':
    main()
