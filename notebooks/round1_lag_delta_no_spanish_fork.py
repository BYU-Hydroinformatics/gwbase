"""
Round-1 revision -- regenerate manuscript Figure 16 (fig:lag_delta_r2_mi) with
Spanish Fork (gage 10152000) dropped entirely.

Same advisor decision as the other round1_*_no_spanish_fork.py scripts. Reproduces
"Figure 3" from notebooks/lag_comparison_analysis.py (the delta-R2/delta-MI bar
chart, one subplot per lag period) using the same computation, filtered to the four
retained catchments. Does not touch the rest of that script's outputs (Figs 1, 2,
4, 5), which are not used in the manuscript.

Run:  ./.venv/bin/python notebooks/round1_lag_delta_no_spanish_fork.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mutual_info_score
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
FEAT    = RESULTS / "features"
OUT_DIR = RESULTS / "round1_revision" / "11_lag_delta_no_spanish_fork"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPANISH_FORK = "10152000"

LAGS = {
    'No Lag':  FEAT / 'data_with_deltas.csv',
    '3 Month': FEAT / 'data_lag_3mo.csv',
    '6 Month': FEAT / 'data_lag_6mo.csv',
    '1 Year':  FEAT / 'data_lag_1yr.csv',
    '5 Year':  FEAT / 'data_lag_5yr.csv',
}
LAG_ORDER = list(LAGS.keys())

LAG_X_COL = {
    'No Lag':  'delta_wte',
    '3 Month': 'delta_wte_lag_3_months',
    '6 Month': 'delta_wte_lag_6_months',
    '1 Year':  'delta_wte_lag_1_year',
    '5 Year':  'delta_wte_lag_5_years',
}

GAGE_NAME_MAP = {
    '10126000': 'Bear River',
    '10141000': 'Weber River',
    '10163000': 'Provo River',
    '10168000': 'Little Cottonwood',
}
RETAINED_ORDER = ['10126000', '10141000', '10163000', '10168000']


def _mi(x, y, n_bins=20):
    x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
    y_bins = np.linspace(y.min(), y.max(), n_bins + 1)
    xd = np.clip(np.digitize(x, x_bins) - 1, 0, n_bins - 1)
    yd = np.clip(np.digitize(y, y_bins) - 1, 0, n_bins - 1)
    return mutual_info_score(xd, yd)


def _reg(x, y):
    if len(x) < 2 or x.std() == 0:
        return dict(slope=np.nan, r2=np.nan, p=np.nan, n=len(x))
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return dict(slope=slope, r2=r ** 2, p=p, n=len(x))


print("Computing regression and MI for each lag (Spanish Fork dropped) …")
gage_records = []
for lag_label, path in LAGS.items():
    x_col = LAG_X_COL[lag_label]
    df = pd.read_csv(path)
    df['gage_id'] = df['gage_id'].astype(str)
    df = df[df['gage_id'] != SPANISH_FORK]
    df = df.dropna(subset=[x_col, 'delta_q'])
    df = df[~((df['gage_id'] == '10163000') & (df[x_col].abs() > 1400))]

    for gage_id, g in df.groupby('gage_id'):
        x = g[x_col].values
        y = g['delta_q'].values
        reg = _reg(x, y)
        mi = _mi(x, y)
        gage_records.append(dict(
            lag=lag_label, gage_id=gage_id, gage_name=GAGE_NAME_MAP.get(gage_id, gage_id),
            n_obs=reg['n'], n_wells=g['well_id'].nunique(),
            **{k: reg[k] for k in ('slope', 'r2', 'p')}, mi=mi
        ))

gage_df = pd.DataFrame(gage_records)
gage_df.to_csv(OUT_DIR / 'lag_comparison_by_gage_no_spanish_fork.csv', index=False)

gages = RETAINED_ORDER
n_gages = len(gages)

no_lag_sub = gage_df[gage_df['lag'] == 'No Lag'].set_index('gage_id')
lag_comparisons = LAG_ORDER[1:]  # ['3 Month', '6 Month', '1 Year', '5 Year']
n_comp = len(lag_comparisons)

fig, axes = plt.subplots(1, n_comp, figsize=(4.5 * n_comp, max(4, n_gages * 0.55 + 1.5)))
if n_comp == 1:
    axes = [axes]

for ax, lag in zip(axes, lag_comparisons):
    lag_sub = gage_df[gage_df['lag'] == lag].set_index('gage_id')

    delta_r2, delta_mi, labels = [], [], []
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
    ax.set_title(lag, fontsize=11)
    ax.set_xlabel('Δ (lag − no lag)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.35)
    ax.set_axisbelow(True)

all_xlims = [ax.get_xlim() for ax in axes]
x_min = min(lim[0] for lim in all_xlims)
x_max = max(lim[1] for lim in all_xlims)
for ax in axes:
    ax.set_xlim(x_min, x_max)

plt.tight_layout()
out = OUT_DIR / 'lag_delta_r2_mi.png'
fig.savefig(out, dpi=600, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

print("\nPeak ΔMI per gage (across the four lags):")
for g in gages:
    sub = gage_df[gage_df['gage_id'] == g]
    no_lag_mi = no_lag_sub.loc[g, 'mi']
    deltas = {lag: sub[sub['lag'] == lag]['mi'].values[0] - no_lag_mi
              for lag in lag_comparisons if lag in sub['lag'].values}
    peak_lag = max(deltas, key=deltas.get)
    print(f"  {GAGE_NAME_MAP.get(g,g):20s} peak ΔMI = {deltas[peak_lag]:+.4f} at {peak_lag}")

print("\nMax ΔR² across all gages/lags:")
r2_deltas = []
for g in gages:
    no_lag_r2 = no_lag_sub.loc[g, 'r2']
    for lag in lag_comparisons:
        sub = gage_df[(gage_df['gage_id'] == g) & (gage_df['lag'] == lag)]
        if len(sub):
            r2_deltas.append((GAGE_NAME_MAP.get(g, g), lag, sub['r2'].values[0] - no_lag_r2))
r2_deltas.sort(key=lambda t: -t[2])
print(f"  max: {r2_deltas[0]}")
