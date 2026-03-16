#!/usr/bin/env python3
"""
Compare results across multiple GWBASE filter scenarios.

Usage:
    python compare_runs.py
    python compare_runs.py --runs filter_pt10_yr3 filter_pt10_yr5 filter_pt10_yr10
    python compare_runs.py --output report/my_comparison
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# ── Default configuration ────────────────────────────────────────────────────

DEFAULT_RUNS = [
    'filter_pt10_yr3',
    'filter_pt10_yr5',
    'filter_pt10_yr10',
]

RUN_LABELS = {
    'filter_none':      'No filter',
    'filter_min10_yr3': 'n≥10, y≥3',
    'filter_min30_yr10':'n≥30, y≥10',
    'filter_pt10_yr3':  'n≥10, y≥3',
    'filter_pt10_yr5':  'n≥10, y≥5',
    'filter_pt10_yr10': 'n≥10, y≥10',
    'run_1':            'n≥20, y≥5 (default)',
}

GAGE_ORDER = ['10126000', '10141000', '10168000']

COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']


# ── Helpers ──────────────────────────────────────────────────────────────────

def label(run: str) -> str:
    return RUN_LABELS.get(run, run)


def load_run(base: str, run: str) -> dict:
    """Load all result files for one run. Returns dict of DataFrames."""
    root = Path(base) / run
    data = {'run': run, 'label': label(run)}

    def _read(path, **kw):
        p = root / path
        return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()

    data['step4']    = _read('processed/well_ts_cleaned.csv')
    data['step5']    = _read('processed/well_pchip_monthly.csv')
    data['step6']    = _read('processed/filtered_by_elevation.csv')
    data['paired']   = _read('processed/paired_well_streamflow.csv')
    data['deltas']   = _read('features/data_with_deltas.csv')
    data['reg_gage'] = _read('features/regression_by_gage.csv')
    data['reg_well'] = _read('features/regression_by_well.csv')
    data['mi']       = _read('features/mi_analysis.csv')
    data['seasonal'] = _read('features/seasonal_analysis.csv')
    data['monthly']  = _read('features/monthly_analysis.csv')
    data['combined'] = _read('features/regression_summary_combined.csv')
    data['mk_well']  = _read('features/mk_well_wte.csv')
    data['mk_gage_wte'] = _read('features/mk_gage_wte.csv')
    data['mk_sf']    = _read('features/mk_streamflow.csv')

    # Well counts funnel
    data['n_step4'] = data['step4']['well_id'].nunique() if len(data['step4']) else 0
    data['n_step5'] = data['step5']['well_id'].nunique() if len(data['step5']) else 0
    data['n_step6'] = data['step6']['well_id'].nunique() if len(data['step6']) else 0
    data['n_paired'] = data['paired']['well_id'].nunique() if len(data['paired']) else 0

    return data


def gage_name(df: pd.DataFrame, gage_id) -> str:
    if 'gage_name' in df.columns:
        row = df[df['gage_id'].astype(str) == str(gage_id)]
        if len(row):
            return str(row.iloc[0]['gage_name'])
    return str(gage_id)


# ── Table builders ───────────────────────────────────────────────────────────

def build_funnel_table(runs_data: list) -> pd.DataFrame:
    rows = []
    for d in runs_data:
        rows.append({
            'Filter scenario': d['label'],
            'Step 4 (cleaned)': d['n_step4'],
            'Step 5 (interpolated)': d['n_step5'],
            'Step 6 (elev. filter)': d['n_step6'],
            'Step 7 (paired)': d['n_paired'],
        })
    return pd.DataFrame(rows)


def build_regression_table(runs_data: list) -> pd.DataFrame:
    """Overall regression by gage across runs."""
    rows = []
    for d in runs_data:
        df = d['reg_gage']
        if df.empty:
            continue
        for _, row in df.iterrows():
            rows.append({
                'Filter scenario': d['label'],
                'gage_id': row['gage_id'],
                'gage_name': row.get('gage_name', row['gage_id']),
                'n_wells': row.get('n_wells', np.nan),
                'n_obs': row.get('n_observations', np.nan),
                'slope': round(row['slope'], 4),
                'R²': round(row['r_squared'], 4),
                'p_value': f"{row['p_value']:.3e}",
            })
    return pd.DataFrame(rows)


def build_mi_table(runs_data: list) -> pd.DataFrame:
    """MI summary by gage across runs."""
    rows = []
    for d in runs_data:
        df = d['mi']
        if df.empty:
            continue
        for gid, grp in df.groupby('gage_id'):
            rows.append({
                'Filter scenario': d['label'],
                'gage_id': gid,
                'gage_name': grp['gage_name'].iloc[0] if 'gage_name' in grp.columns else gid,
                'n_wells': len(grp),
                'MI mean': round(grp['mi'].mean(), 4),
                'MI median': round(grp['mi'].median(), 4),
                'Pearson r mean': round(grp['pearson_r'].mean(), 4),
                'Spearman r mean': round(grp['spearman_r'].mean(), 4),
            })
    return pd.DataFrame(rows)


def build_seasonal_table(runs_data: list) -> pd.DataFrame:
    rows = []
    for d in runs_data:
        df = d['seasonal']
        if df.empty:
            continue
        period_col = 'period' if 'period' in df.columns else 'season'
        for _, row in df.iterrows():
            rows.append({
                'Filter scenario': d['label'],
                'gage_id': row['gage_id'],
                'gage_name': row.get('gage_name', row['gage_id']),
                'season': row.get(period_col, ''),
                'n_obs': row.get('n_observations', np.nan),
                'slope': round(row['slope'], 4),
                'R²': round(row['r_squared'], 4),
                'p_value': f"{row['p_value']:.3e}",
            })
    return pd.DataFrame(rows)


# ── Plot builders ────────────────────────────────────────────────────────────

def plot_funnel(runs_data: list, out_dir: str):
    """Bar chart: well count funnel across scenarios."""
    steps = ['Step 4\n(cleaned)', 'Step 5\n(interp.)', 'Step 6\n(elev.)', 'Step 7\n(paired)']
    keys  = ['n_step4', 'n_step5', 'n_step6', 'n_paired']
    n = len(runs_data)
    x = np.arange(len(steps))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, d in enumerate(runs_data):
        vals = [d[k] for k in keys]
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=d['label'], color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(v), ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(steps)
    ax.set_ylabel('Number of Wells')
    ax.set_title('Well Count Funnel by Filter Scenario')
    ax.legend(loc='upper right', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'funnel_well_counts.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_regression_r2_by_gage(runs_data: list, out_dir: str):
    """Grouped bar: R² by gage and scenario."""
    gages = GAGE_ORDER
    n_runs = len(runs_data)
    x = np.arange(len(gages))
    width = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, d in enumerate(runs_data):
        df = d['reg_gage']
        if df.empty:
            continue
        df['gage_id'] = df['gage_id'].astype(str)
        vals = [df[df['gage_id'] == g]['r_squared'].values[0] if g in df['gage_id'].values else 0 for g in gages]
        offset = (i - n_runs/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=d['label'], color=COLORS[i % len(COLORS)], alpha=0.85)

    # Get gage names from first run with data
    x_labels = []
    for g in gages:
        name = g
        for d in runs_data:
            df = d['reg_gage']
            if not df.empty and g in df['gage_id'].astype(str).values:
                name = df[df['gage_id'].astype(str) == g]['gage_name'].values[0] if 'gage_name' in df.columns else g
                break
        x_labels.append(f"{g}\n{name[:30]}")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel('R²')
    ax.set_title('Overall Regression R² (ΔWTE vs ΔQ) by Gage')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'regression_r2_by_gage.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_mi_by_gage(runs_data: list, out_dir: str):
    """Grouped bar: mean MI by gage and scenario (with Pearson r as line)."""
    gages = GAGE_ORDER
    n_runs = len(runs_data)
    x = np.arange(len(gages))
    width = 0.8 / n_runs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel in [
        (axes[0], 'mi', 'Mean MI'),
        (axes[1], 'pearson_r', 'Mean |Pearson r|'),
    ]:
        for i, d in enumerate(runs_data):
            df = d['mi']
            if df.empty:
                continue
            df = df.copy()
            df['gage_id'] = df['gage_id'].astype(str)
            vals = []
            for g in gages:
                grp = df[df['gage_id'] == g]
                if len(grp):
                    v = grp[metric].mean() if metric == 'mi' else grp[metric].abs().mean()
                else:
                    v = 0
                vals.append(v)
            offset = (i - n_runs/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=d['label'], color=COLORS[i % len(COLORS)], alpha=0.85)

        x_labels = []
        for g in gages:
            name = g
            for d in runs_data:
                df = d['mi']
                if not df.empty and g in df['gage_id'].astype(str).values:
                    name = df[df['gage_id'].astype(str) == g]['gage_name'].values[0] if 'gage_name' in df.columns else g
                    break
            x_labels.append(f"{g}\n{name[:25]}")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ' by Gage')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mi_by_gage.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_mi_distribution(runs_data: list, out_dir: str):
    """Box plot: MI distribution per scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax_idx, g in enumerate(GAGE_ORDER):
        ax = axes[ax_idx]
        data_list, labels_list = [], []
        for d in runs_data:
            df = d['mi']
            if df.empty:
                continue
            df = df.copy()
            df['gage_id'] = df['gage_id'].astype(str)
            grp = df[df['gage_id'] == g]['mi']
            if len(grp):
                data_list.append(grp.values)
                labels_list.append(d['label'])

        if data_list:
            bp = ax.boxplot(data_list, tick_labels=labels_list, patch_artist=True, notch=False)
            for patch, color in zip(bp['boxes'], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        # Get gage name
        gname = g
        for d in runs_data:
            df = d['mi']
            if not df.empty and g in df['gage_id'].astype(str).values:
                gname = df[df['gage_id'].astype(str) == g]['gage_name'].values[0] if 'gage_name' in df.columns else g
                break

        ax.set_title(f'{g}\n{gname[:35]}', fontsize=9)
        ax.set_ylabel('MI')
        ax.tick_params(axis='x', rotation=15, labelsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('MI Distribution by Gage and Filter Scenario', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mi_distribution_by_gage.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_seasonal_r2(runs_data: list, out_dir: str):
    """Heatmap-style: R² by season and gage for each scenario."""
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    n_runs = len(runs_data)
    fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 4), sharey=True)
    if n_runs == 1:
        axes = [axes]

    vmax = 0
    all_mats = []
    for d in runs_data:
        df = d['seasonal']
        if df.empty:
            all_mats.append(None)
            continue
        df['gage_id'] = df['gage_id'].astype(str)
        mat = pd.DataFrame(index=GAGE_ORDER, columns=seasons, dtype=float)
        period_col = 'period' if 'period' in df.columns else 'season'
        for g in GAGE_ORDER:
            for s in seasons:
                rows = df[(df['gage_id'] == g) & (df[period_col] == s)]
                if len(rows):
                    mat.loc[g, s] = rows['r_squared'].values[0]
        all_mats.append(mat)
        vmax = max(vmax, mat.values.astype(float).max() if not mat.isnull().all().all() else 0)

    vmax = max(vmax, 0.01)

    for ax, d, mat in zip(axes, runs_data, all_mats):
        if mat is None:
            ax.set_title(d['label'])
            continue
        im = ax.imshow(mat.values.astype(float), aspect='auto', cmap='YlOrRd',
                       vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_xticks(range(len(seasons)))
        ax.set_xticklabels(seasons, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(GAGE_ORDER)))
        ax.set_yticklabels(GAGE_ORDER, fontsize=8)
        ax.set_title(d['label'], fontsize=9)
        for i, g in enumerate(GAGE_ORDER):
            for j, s in enumerate(seasons):
                val = mat.loc[g, s]
                if not pd.isna(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7,
                            color='black' if val < vmax * 0.6 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.8, label='R²')

    plt.suptitle('Seasonal Regression R² by Gage and Filter Scenario', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'seasonal_r2_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_well_count_per_gage(runs_data: list, out_dir: str):
    """Grouped bar: paired well count per gage per scenario."""
    gages = GAGE_ORDER
    n_runs = len(runs_data)
    x = np.arange(len(gages))
    width = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, d in enumerate(runs_data):
        df = d['paired']
        if df.empty:
            continue
        df = df.copy()
        df['gage_id'] = df['gage_id'].astype(str)
        vals = [df[df['gage_id'] == g]['well_id'].nunique() for g in gages]
        offset = (i - n_runs/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=d['label'], color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(v), ha='center', va='bottom', fontsize=8)

    x_labels = []
    for g in gages:
        name = g
        for d in runs_data:
            df = d['paired']
            if not df.empty and g in df['gage_id'].astype(str).values:
                name = df[df['gage_id'].astype(str) == g]['gage_name'].values[0] if 'gage_name' in df.columns else g
                break
        x_labels.append(f"{g}\n{name[:28]}")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel('Number of Paired Wells')
    ax.set_title('Paired Well Count by Gage and Filter Scenario')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'paired_wells_by_gage.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_slope_comparison(runs_data: list, out_dir: str):
    """Overall regression slope comparison across runs and gages."""
    gages = GAGE_ORDER
    n_runs = len(runs_data)
    x = np.arange(len(gages))
    width = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, d in enumerate(runs_data):
        df = d['reg_gage']
        if df.empty:
            continue
        df['gage_id'] = df['gage_id'].astype(str)
        vals = [df[df['gage_id'] == g]['slope'].values[0] if g in df['gage_id'].values else 0 for g in gages]
        offset = (i - n_runs/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=d['label'], color=COLORS[i % len(COLORS)], alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    x_labels = []
    for g in gages:
        name = g
        for d in runs_data:
            df = d['reg_gage']
            if not df.empty and g in df['gage_id'].astype(str).values:
                name = df[df['gage_id'].astype(str) == g]['gage_name'].values[0] if 'gage_name' in df.columns else g
                break
        x_labels.append(f"{g}\n{name[:28]}")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel('Slope (ΔQ / ΔWTE, cfs/ft)')
    ax.set_title('Overall Regression Slope by Gage and Filter Scenario')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'regression_slope_by_gage.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ── MK + Sen's Slope tables & plots ─────────────────────────────────────────

def build_mk_well_summary(runs_data: list) -> pd.DataFrame:
    """Summary of per-well MK results across scenarios."""
    rows = []
    for d in runs_data:
        df = d['mk_well']
        if df.empty:
            continue
        sig = (df['p_value'] < 0.05).sum()
        dec = ((df['p_value'] < 0.05) & (df['sen_slope_yr'] < 0)).sum()
        inc = ((df['p_value'] < 0.05) & (df['sen_slope_yr'] > 0)).sum()
        rows.append({
            'Filter scenario': d['label'],
            'Wells tested': len(df),
            'Sig. (p<0.05)': sig,
            'Sig. %': round(sig / len(df) * 100, 1),
            'Declining': dec,
            'Increasing': inc,
            'Median slope (ft/yr)': round(df['sen_slope_yr'].median(), 4),
            'Mean slope (ft/yr)': round(df['sen_slope_yr'].mean(), 4),
        })
    return pd.DataFrame(rows)


def build_mk_gage_table(runs_data: list, mk_key: str, slope_unit: str) -> pd.DataFrame:
    rows = []
    for d in runs_data:
        df = d[mk_key]
        if df.empty:
            continue
        for _, row in df.iterrows():
            rows.append({
                'Filter scenario': d['label'],
                'gage_id': row['gage_id'],
                'gage_name': row.get('gage_name', row['gage_id']),
                'n_obs': row.get('n_months', row.get('n_obs', np.nan)),
                'trend': row['trend'],
                'h': row['h'],
                'p_value': f"{row['p_value']:.3e}",
                'tau': round(row['tau'], 4),
                f'sen_slope ({slope_unit})': round(row['sen_slope_yr'], 4),
            })
    return pd.DataFrame(rows)


def plot_mk_well_sen_slope(runs_data: list, out_dir: str):
    """Box plot of Sen's slope (ft/yr) per well, grouped by scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax_idx, g in enumerate(GAGE_ORDER):
        ax = axes[ax_idx]
        data_list, labels_list = [], []
        for d in runs_data:
            df = d['mk_well']
            if df.empty or 'gage_id' not in df.columns:
                continue
            df = df.copy()
            df['gage_id'] = df['gage_id'].astype(str)
            grp = df[df['gage_id'] == g]['sen_slope_yr'].dropna()
            if len(grp):
                data_list.append(grp.values)
                labels_list.append(d['label'])

        if data_list:
            bp = ax.boxplot(data_list, tick_labels=labels_list, patch_artist=True,
                            notch=False, showfliers=False)
            for patch, color in zip(bp['boxes'], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.7)

        gname = g
        for d in runs_data:
            df = d['mk_well']
            if not df.empty and 'gage_id' in df.columns and g in df['gage_id'].astype(str).values:
                if 'gage_name' in df.columns:
                    gname = df[df['gage_id'].astype(str) == g]['gage_name'].iloc[0]
                break

        ax.set_title(f'{g}\n{str(gname)[:35]}', fontsize=9)
        ax.set_ylabel("Sen's slope (ft/yr)")
        ax.tick_params(axis='x', rotation=15, labelsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Per-well Sen's Slope Distribution (WTE, ft/yr)\n(outliers hidden; red dashed = 0)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mk_well_sen_slope.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_mk_well_trend_pct(runs_data: list, out_dir: str):
    """Stacked bar: % declining / no-trend / increasing wells per gage per scenario."""
    gages = GAGE_ORDER
    n_runs = len(runs_data)
    x = np.arange(len(gages))
    width = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, d in enumerate(runs_data):
        df = d['mk_well']
        if df.empty or 'gage_id' not in df.columns:
            continue
        df = df.copy()
        df['gage_id'] = df['gage_id'].astype(str)
        offset = (i - n_runs / 2 + 0.5) * width
        bottoms = np.zeros(len(gages))
        for color, mask_fn, lbl in [
            ('#EF5350', lambda g: df[(df['gage_id'] == g) & (df['p_value'] < 0.05) & (df['sen_slope_yr'] < 0)], 'Declining*'),
            ('#BDBDBD', lambda g: df[(df['gage_id'] == g) & (df['p_value'] >= 0.05)], 'No trend'),
            ('#42A5F5', lambda g: df[(df['gage_id'] == g) & (df['p_value'] < 0.05) & (df['sen_slope_yr'] > 0)], 'Increasing*'),
        ]:
            vals = []
            for g in gages:
                total = len(df[df['gage_id'] == g])
                pct = len(mask_fn(g)) / total * 100 if total > 0 else 0
                vals.append(pct)
            bar_lbl = lbl if i == 0 else '_'
            ax.bar(x + offset, vals, width, bottom=bottoms, color=color,
                   label=bar_lbl, alpha=0.85)
            bottoms = bottoms + np.array(vals)

    x_labels = []
    for g in gages:
        name = g
        for d in runs_data:
            df = d['mk_well']
            if not df.empty and 'gage_id' in df.columns and g in df['gage_id'].astype(str).values:
                if 'gage_name' in df.columns:
                    name = df[df['gage_id'].astype(str) == g]['gage_name'].iloc[0]
                break
        x_labels.append(f"{g}\n{str(name)[:28]}")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel('% of wells')
    ax.set_ylim(0, 105)
    ax.set_title('Per-well WTE Trend by Gage (* p<0.05)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add scenario labels inside bars
    for i, d in enumerate(runs_data):
        offset = (i - n_runs / 2 + 0.5) * width
        ax.text(x[0] + offset, 102, d['label'], ha='center', va='bottom',
                fontsize=6, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mk_well_trend_pct.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_mk_gage_streamflow(runs_data: list, out_dir: str):
    """Bar: gage streamflow Sen's slope (cfs/yr) — same for all scenarios."""
    # Streamflow MK is independent of well filter, so use first run with data
    df = None
    for d in runs_data:
        if not d['mk_sf'].empty:
            df = d['mk_sf'].copy()
            break
    if df is None:
        return

    df['gage_id'] = df['gage_id'].astype(str)
    df = df.sort_values('sen_slope_yr')
    colors = ['#EF5350' if v < 0 else '#42A5F5' for v in df['sen_slope_yr']]
    sig_marker = ['*' if h else '' for h in df['h']]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(df)), df['sen_slope_yr'], color=colors, alpha=0.85)
    labels = [f"{row['gage_id']} {row.get('gage_name','')[:30]}{sig_marker[i]}"
              for i, (_, row) in enumerate(df.iterrows())]
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Sen's slope (cfs/yr)")
    ax.set_title("Streamflow Trend — Sen's Slope per Gage\n(* p<0.05, red=declining, blue=increasing)")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mk_streamflow_slope.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_mk_gage_wte(runs_data: list, out_dir: str):
    """Grouped bar: gage-aggregated WTE Sen's slope across scenarios."""
    gages = GAGE_ORDER
    n_runs = len(runs_data)
    x = np.arange(len(gages))
    width = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, d in enumerate(runs_data):
        df = d['mk_gage_wte']
        if df.empty:
            continue
        df = df.copy()
        df['gage_id'] = df['gage_id'].astype(str)
        vals = [df[df['gage_id'] == g]['sen_slope_yr'].values[0]
                if g in df['gage_id'].values else 0 for g in gages]
        sigs = [bool(df[df['gage_id'] == g]['h'].values[0])
                if g in df['gage_id'].values else False for g in gages]
        offset = (i - n_runs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=d['label'],
                      color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar, v, sig in zip(bars, vals, sigs):
            if sig:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + (0.02 if v >= 0 else -0.08),
                        '*', ha='center', fontsize=11, color='black')

    x_labels = []
    for g in gages:
        name = g
        for d in runs_data:
            df = d['mk_gage_wte']
            if not df.empty and g in df['gage_id'].astype(str).values:
                if 'gage_name' in df.columns:
                    name = df[df['gage_id'].astype(str) == g]['gage_name'].iloc[0]
                break
        x_labels.append(f"{g}\n{str(name)[:28]}")

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Sen's slope (ft/yr)")
    ax.set_title("Catchment-mean WTE Trend — Sen's Slope by Gage (* p<0.05)")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mk_gage_wte_slope.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', default=DEFAULT_RUNS)
    parser.add_argument('--base', default='report')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    out_dir = args.output or os.path.join(args.base, 'comparison_' + '_'.join(args.runs))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load all runs
    runs_data = []
    for run in args.runs:
        run_path = os.path.join(args.base, run)
        if not os.path.isdir(run_path):
            print(f"  WARNING: {run_path} not found, skipping.")
            continue
        d = load_run(args.base, run)
        runs_data.append(d)
        print(f"  Loaded {run}: step4={d['n_step4']}, step6={d['n_step6']}, paired={d['n_paired']}")

    if not runs_data:
        print("No valid runs found.")
        sys.exit(1)

    print(f"\nGenerating comparison for {len(runs_data)} runs...")

    # ── Tables ──
    funnel = build_funnel_table(runs_data)
    funnel.to_csv(os.path.join(out_dir, 'table_well_funnel.csv'), index=False)
    print("  table_well_funnel.csv")

    reg = build_regression_table(runs_data)
    reg.to_csv(os.path.join(out_dir, 'table_regression_by_gage.csv'), index=False)
    print("  table_regression_by_gage.csv")

    mi = build_mi_table(runs_data)
    mi.to_csv(os.path.join(out_dir, 'table_mi_by_gage.csv'), index=False)
    print("  table_mi_by_gage.csv")

    seas = build_seasonal_table(runs_data)
    seas.to_csv(os.path.join(out_dir, 'table_seasonal_regression.csv'), index=False)
    print("  table_seasonal_regression.csv")

    # ── Plots ──
    plot_funnel(runs_data, out_dir)
    print("  funnel_well_counts.png")

    plot_well_count_per_gage(runs_data, out_dir)
    print("  paired_wells_by_gage.png")

    plot_regression_r2_by_gage(runs_data, out_dir)
    print("  regression_r2_by_gage.png")

    plot_slope_comparison(runs_data, out_dir)
    print("  regression_slope_by_gage.png")

    plot_mi_by_gage(runs_data, out_dir)
    print("  mi_by_gage.png")

    plot_mi_distribution(runs_data, out_dir)
    print("  mi_distribution_by_gage.png")

    plot_seasonal_r2(runs_data, out_dir)
    print("  seasonal_r2_heatmap.png")

    # ── MK + Sen's Slope ──
    mk_well_summary = build_mk_well_summary(runs_data)
    mk_well_summary.to_csv(os.path.join(out_dir, 'table_mk_well_summary.csv'), index=False)
    print("  table_mk_well_summary.csv")

    mk_gage_wte_table = build_mk_gage_table(runs_data, 'mk_gage_wte', 'ft/yr')
    mk_gage_wte_table.to_csv(os.path.join(out_dir, 'table_mk_gage_wte.csv'), index=False)
    print("  table_mk_gage_wte.csv")

    mk_sf_table = build_mk_gage_table(runs_data, 'mk_sf', 'cfs/yr')
    mk_sf_table.to_csv(os.path.join(out_dir, 'table_mk_streamflow.csv'), index=False)
    print("  table_mk_streamflow.csv")

    plot_mk_well_sen_slope(runs_data, out_dir)
    print("  mk_well_sen_slope.png")

    plot_mk_well_trend_pct(runs_data, out_dir)
    print("  mk_well_trend_pct.png")

    plot_mk_gage_wte(runs_data, out_dir)
    print("  mk_gage_wte_slope.png")

    plot_mk_gage_streamflow(runs_data, out_dir)
    print("  mk_streamflow_slope.png")

    # ── Print summary table to console ──
    print("\n" + "="*70)
    print("FUNNEL SUMMARY")
    print("="*70)
    print(funnel.to_string(index=False))

    print("\n" + "="*70)
    print("REGRESSION BY GAGE")
    print("="*70)
    print(reg[['Filter scenario','gage_name','n_wells','n_obs','slope','R²','p_value']].to_string(index=False))

    print("\n" + "="*70)
    print("MI BY GAGE (mean)")
    print("="*70)
    print(mi[['Filter scenario','gage_name','n_wells','MI mean','Pearson r mean']].to_string(index=False))

    print("\n" + "="*70)
    print("MK TEST — WELL WTE TREND SUMMARY")
    print("="*70)
    print(mk_well_summary.to_string(index=False))

    print("\n" + "="*70)
    print("MK TEST — GAGE-AGGREGATED WTE TREND")
    print("="*70)
    print(mk_gage_wte_table[['Filter scenario','gage_name','trend','h','p_value','tau',
                              'sen_slope (ft/yr)']].to_string(index=False))

    print("\n" + "="*70)
    print("MK TEST — STREAMFLOW TREND (first run, independent of filter)")
    print("="*70)
    sf_first = mk_sf_table[mk_sf_table['Filter scenario'] == runs_data[0]['label']]
    print(sf_first[['gage_name','trend','h','p_value','tau',
                    'sen_slope (cfs/yr)']].to_string(index=False))

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == '__main__':
    main()
