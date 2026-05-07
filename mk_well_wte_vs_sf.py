#!/usr/bin/env python3
"""
Per-well MK + Sen's slope for WTE, plus gage streamflow Sen's slope.

Two figures:
  1. Full record  – box plot of well Sen's slope per basin, 4 filter scenarios
  2. Last 50 yr   – same but data & filters applied to 1976+ only

Filtering for last-50yr uses the same min_pts / min_yrs criteria applied to
the truncated (1976+) data.  No hard MIN_OBS cap is added.

Output:
  report/comparison_pts_yr3/mk_well_vs_sf_fullrecord.png
  report/comparison_pts_yr3/mk_well_vs_sf_last50yr.png
  report/comparison_pts_yr3/mk_well_vs_sf_table.csv
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
    (2,  3, 'n≥2, y≥3'),
    (10, 3, 'n≥10, y≥3'),
    (20, 3, 'n≥20, y≥3'),
    (30, 3, 'n≥30, y≥3'),
]
REPORT_BASE       = 'report'
BASE_SCENARIO_DIR = 'filter_pt2_yr3'   # most permissive → base pool of wells
SF_DIR            = 'filter_pt10_yr3'  # any run shares the same streamflow file
OUT_DIR           = 'report/comparison_pts_yr3'
CUTOFF_50YR       = pd.Timestamp('1976-01-01')

GAGE_ORDER = ['10126000', '10141000', '10168000']
GAGE_NAMES = {
    '10126000': 'Bear River nr Corinne',
    '10141000': 'Weber River nr Plain City',
    '10168000': 'Little Cottonwood Cr',
}
COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']


# ── Helpers ──────────────────────────────────────────────────────────────────

def filter_wells_in_period(df: pd.DataFrame, min_pts: int, min_yrs: float) -> pd.DataFrame:
    """Keep wells that satisfy min_pts observations AND min_yrs span within df."""
    stats = (df.groupby('well_id')
               .agg(n=('wte', 'count'),
                    d0=('date', 'min'),
                    d1=('date', 'max'))
               .reset_index())
    stats['yr_span'] = (stats['d1'] - stats['d0']).dt.days / 365.25
    valid = stats.loc[(stats['n'] >= min_pts) & (stats['yr_span'] >= min_yrs), 'well_id']
    return df[df['well_id'].isin(valid)]


def run_mk(series: pd.Series, dates: pd.Series):
    """MK test + Sen's slope (per yr). Returns dict or None."""
    vals = series.dropna().values
    if len(vals) < 4:
        return None
    try:
        res = mk.original_test(vals)
    except Exception:
        return None
    intervals = dates.diff().dt.days.dropna()
    med_days = intervals.median()
    slope_yr = res.slope * (365.25 / med_days) if med_days > 0 else np.nan
    return {
        'trend': res.trend,
        'h': bool(res.h),
        'p_value': float(res.p),
        'tau': float(res.Tau),
        'sen_slope': float(res.slope),
        'sen_slope_yr': slope_yr,
        'n': len(vals),
    }


def per_well_mk(df: pd.DataFrame) -> pd.DataFrame:
    """Run MK on each well in df. Returns DataFrame with one row per well."""
    rows = []
    for wid, grp in df.groupby('well_id'):
        grp = grp.dropna(subset=['wte']).sort_values('date')
        r = run_mk(grp['wte'], grp['date'])
        if r is None:
            continue
        r['well_id'] = wid
        r['gage_id'] = str(grp['gage_id'].iloc[0])
        r['gage_name'] = grp['gage_name'].iloc[0] if 'gage_name' in grp.columns else ''
        rows.append(r)
    return pd.DataFrame(rows)


def sf_mk_for_period(sf: pd.DataFrame, cutoff=None):
    """Compute MK + Sen's slope on streamflow per gage, optionally after cutoff."""
    if cutoff is not None:
        sf = sf[sf['date'] >= cutoff]
    results = {}
    for gid, grp in sf.groupby('gage_id'):
        grp = grp.sort_values('date').dropna(subset=['q'])
        r = run_mk(grp['q'], grp['date'])
        results[str(gid)] = r
    return results


def compute_period(base_df: pd.DataFrame, cutoff=None):
    """
    For each scenario, filter wells within the period and run per-well MK.
    Returns list of (label, well_mk_df).
    """
    df = base_df.copy()
    if cutoff is not None:
        df = df[df['date'] >= cutoff]

    results = []
    for min_pts, min_yrs, label in SCENARIOS:
        filtered = filter_wells_in_period(df, min_pts, min_yrs)
        well_mk = per_well_mk(filtered)
        n_wells = well_mk['well_id'].nunique() if not well_mk.empty else 0
        print(f"  {label}: {n_wells} wells")
        results.append((label, well_mk))
    return results


def make_figure(period_results, sf_results, title_suffix, out_path):
    """
    Box plot: 3 subplots (one per gage), 4 boxes (one per scenario).
    Streamflow Sen's slope annotated as text box.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax_idx, gid in enumerate(GAGE_ORDER):
        ax = axes[ax_idx]
        data_list, labels_list = [], []

        for label, well_mk in period_results:
            if well_mk.empty or 'gage_id' not in well_mk.columns:
                data_list.append(np.array([]))
                labels_list.append(label)
                continue
            grp = well_mk[well_mk['gage_id'] == gid]['sen_slope_yr'].dropna()
            data_list.append(grp.values)
            labels_list.append(f"{label}\n(n={len(grp)})")

        nonempty = [d for d in data_list if len(d) > 0]
        if nonempty:
            bp = ax.boxplot([d for d in data_list],
                            tick_labels=labels_list,
                            patch_artist=True, notch=False, showfliers=False)
            for patch, color in zip(bp['boxes'], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.6)
        ax.set_title(GAGE_NAMES.get(gid, gid), fontsize=9, fontweight='bold')
        ax.set_ylabel("WTE Sen's slope (ft/yr)")
        ax.tick_params(axis='x', labelsize=7.5)
        ax.grid(axis='y', alpha=0.3)

        # Streamflow annotation
        sf_r = sf_results.get(gid)
        if sf_r:
            p_str = f"{sf_r['p_value']:.3f}" if sf_r['p_value'] >= 0.001 else "< 0.001"
            sig_str = '*' if sf_r['p_value'] < 0.05 else ''
            sf_txt = (f"Q Sen's slope: {sf_r['sen_slope_yr']:+.3f} cfs/yr{sig_str}\n"
                      f"p = {p_str}  (n={sf_r['n']})")
            ax.text(0.98, 0.97, sf_txt,
                    transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              edgecolor='#aaa', alpha=0.9))

    plt.suptitle(
        f"Per-well WTE Sen's Slope Distribution — {title_suffix}\n"
        "(box = IQR, whiskers = 5–95th pct; outliers hidden; red dashed = 0)",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load base well data (most permissive filter on full record)
    base_path = os.path.join(REPORT_BASE, BASE_SCENARIO_DIR, 'features', 'data_with_deltas.csv')
    print(f"Loading base well data from {base_path} ...")
    base_df = pd.read_csv(base_path, parse_dates=['date'])
    base_df['gage_id'] = base_df['gage_id'].astype(str)
    print(f"  {base_df['well_id'].nunique()} unique wells, "
          f"{base_df['date'].min().date()} – {base_df['date'].max().date()}")

    # Load streamflow
    sf_path = os.path.join(REPORT_BASE, SF_DIR, 'processed', 'streamflow_monthly_bfd.csv')
    sf = pd.read_csv(sf_path, parse_dates=['date'])
    sf['gage_id'] = sf['gage_id'].astype(str)

    # ── Full record ──────────────────────────────────────────────────────────
    print("\n=== FULL RECORD ===")
    full_results = compute_period(base_df, cutoff=None)
    sf_full = sf_mk_for_period(sf, cutoff=None)

    # ── Last 50 years ────────────────────────────────────────────────────────
    print(f"\n=== LAST 50 YEARS (>= {CUTOFF_50YR.date()}) ===")
    last50_results = compute_period(base_df, cutoff=CUTOFF_50YR)
    sf_last50 = sf_mk_for_period(sf, cutoff=CUTOFF_50YR)

    # ── Figures ──────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    make_figure(full_results, sf_full,
                "Full Record",
                os.path.join(OUT_DIR, 'mk_well_vs_sf_fullrecord.png'))

    make_figure(last50_results, sf_last50,
                "Last 50 Years (1976+)",
                os.path.join(OUT_DIR, 'mk_well_vs_sf_last50yr.png'))

    # ── Combined table ───────────────────────────────────────────────────────
    rows = []
    for period_label, period_results, sf_res in [
        ('Full record',    full_results,   sf_full),
        ('Last 50yr',      last50_results, sf_last50),
    ]:
        for label, well_mk in period_results:
            if well_mk.empty:
                continue
            for gid in GAGE_ORDER:
                g = well_mk[well_mk['gage_id'] == gid]
                if g.empty:
                    continue
                sig = (g['p_value'] < 0.05).sum()
                dec = ((g['p_value'] < 0.05) & (g['sen_slope_yr'] < 0)).sum()
                inc = ((g['p_value'] < 0.05) & (g['sen_slope_yr'] > 0)).sum()
                sf_r = sf_res.get(gid, {}) or {}
                rows.append({
                    'period': period_label,
                    'filter': label,
                    'gage_id': gid,
                    'gage_name': GAGE_NAMES.get(gid, gid),
                    'n_wells': len(g),
                    'pct_sig': round(sig / len(g) * 100, 1),
                    'pct_declining': round(dec / len(g) * 100, 1),
                    'pct_increasing': round(inc / len(g) * 100, 1),
                    'median_slope_ft_yr': round(g['sen_slope_yr'].median(), 4),
                    'mean_slope_ft_yr':   round(g['sen_slope_yr'].mean(), 4),
                    'sf_slope_cfs_yr': round(sf_r.get('sen_slope_yr', np.nan), 4),
                    'sf_p_value':      round(sf_r.get('p_value', np.nan), 4),
                    'sf_trend':        sf_r.get('trend', ''),
                })

    tbl = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, 'mk_well_vs_sf_table.csv')
    tbl.to_csv(out_csv, index=False)
    print(f"  → {out_csv}")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(tbl.to_string(index=False))


if __name__ == '__main__':
    main()
