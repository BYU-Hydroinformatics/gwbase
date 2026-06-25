"""
WTE and Q Time Series with BFD Periods — real data, polished style.
Well 404024111154501, Gage 10163000 (1997-08 to 2000-09).
No title, dpi=600.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'analysis', 'filter_sensitivity')
os.makedirs(OUT_DIR, exist_ok=True)

WELL_ID = 404024111154501
GAGE_ID = 10163000

# ── Load data ─────────────────────────────────────────────────────────────────
paired = pd.read_csv(
    os.path.join(os.path.dirname(__file__), '..', 'results', 'processed', 'paired_well_streamflow.csv'),
    parse_dates=['date']
)
wte_df = (
    paired[(paired['well_id'] == WELL_ID) & (paired['gage_id'] == GAGE_ID)]
    .sort_values('date').copy()
)
WTE0 = wte_df['wte0'].iloc[0]
Q0   = wte_df['q0'].iloc[0]

q_daily = pd.read_csv(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'streamflow',
                 'gages_with_bfd_predictions', f'{GAGE_ID}.csv'),
    parse_dates=['date']
)
q_daily = q_daily[
    (q_daily['date'] >= wte_df['date'].min()) &
    (q_daily['date'] <= wte_df['date'].max())
].copy()
q_daily['bfd_flag'] = (q_daily['bfd'] == 1)

# Aggregate daily BFD to monthly: month is BFD if ≥50% of days are BFD=1
q_daily['year_month'] = q_daily['date'].dt.to_period('M')
monthly_bfd = (
    q_daily.groupby('year_month')['bfd']
    .apply(lambda x: (x == 1).mean())
    .rename('bfd_frac')
    .reset_index()
)
monthly_bfd['bfd_month'] = monthly_bfd['bfd_frac'] >= 0.5
monthly_bfd['month_start'] = monthly_bfd['year_month'].dt.to_timestamp(how='start')
monthly_bfd['month_end']   = monthly_bfd['year_month'].dt.to_timestamp(how='end')

# Join monthly BFD flag back to WTE and daily Q
wte_df['year_month'] = wte_df['date'].dt.to_period('M')
wte_df = wte_df.merge(monthly_bfd[['year_month', 'bfd_month']], on='year_month', how='left')

q_daily = q_daily.merge(monthly_bfd[['year_month', 'bfd_month']], on='year_month', how='left')

baseline_date = wte_df['date'].iloc[0]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.6,
})

C_BFD_SHADE = '#d6eaf8'
C_WTE_LINE  = '#2c3e50'
C_WTE_DOT   = '#2980b9'
C_Q_LINE    = '#2c3e50'
C_Q_DOT     = '#27ae60'
C_REF_LINE  = '#e67e22'
C_STAR      = '#f1c40f'
C_ANNOTBOX  = dict(boxstyle='round,pad=0.35', facecolor='white',
                   edgecolor='#aaaaaa', lw=0.8, alpha=0.95)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                constrained_layout=True)

# ── BFD shading — month-level so shading aligns exactly with WTE dots ─────────
in_bfd, seg_start = False, None
for _, row in monthly_bfd.iterrows():
    if row['bfd_month'] and not in_bfd:
        seg_start, in_bfd = row['month_start'], True
    elif not row['bfd_month'] and in_bfd:
        for ax in (ax1, ax2):
            ax.axvspan(seg_start, row['month_start'], color=C_BFD_SHADE, lw=0, zorder=0)
        in_bfd = False
if in_bfd:
    for ax in (ax1, ax2):
        ax.axvspan(seg_start, monthly_bfd['month_end'].iloc[-1],
                   color=C_BFD_SHADE, lw=0, zorder=0)

# ── WTE panel ─────────────────────────────────────────────────────────────────
wte_bfd = wte_df[wte_df['bfd_month'] == True]
ax1.plot(wte_df['date'], wte_df['wte'],
         color=C_WTE_LINE, linewidth=1.4, zorder=2, label='WTE')
ax1.scatter(wte_bfd['date'], wte_bfd['wte'],
            color=C_WTE_DOT, s=30, zorder=3, edgecolors='white', linewidths=0.4,
            label='WTE (BFD=1)')
ax1.axhline(WTE0, color=C_REF_LINE, linestyle='--', linewidth=1.2, alpha=0.85, zorder=1)
ax1.plot(baseline_date, WTE0, marker='*', color=C_STAR,
         markersize=15, zorder=5, linestyle='none',
         markeredgecolor='#c0a010', markeredgewidth=0.6)
ax1.annotate(f'{baseline_date.strftime("%Y-%m-%d")}\nWTE₀ = {WTE0:.2f}',
             xy=(baseline_date, WTE0), xytext=(50, 18),
             textcoords='offset points', fontsize=8.5,
             bbox=C_ANNOTBOX,
             arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8))
ax1.set_ylabel('WTE (ft)', fontsize=10)
ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.9, edgecolor='#cccccc')
ax1.tick_params(labelsize=9)

# ── Q panel ───────────────────────────────────────────────────────────────────
q_bfd = q_daily[q_daily['bfd_month'] == True]
ax2.plot(q_daily['date'], q_daily['streamflow'],
         color=C_Q_LINE, linewidth=0.8, zorder=2, label='Q')
ax2.scatter(q_bfd['date'], q_bfd['streamflow'],
            color=C_Q_DOT, s=8, zorder=3, edgecolors='none',
            label='Q (BFD=1)')
ax2.axhline(Q0, color=C_REF_LINE, linestyle='--', linewidth=1.2, alpha=0.85, zorder=1)
ax2.plot(baseline_date, Q0, marker='*', color=C_STAR,
         markersize=15, zorder=5, linestyle='none',
         markeredgecolor='#c0a010', markeredgewidth=0.6)
ax2.annotate(f'{baseline_date.strftime("%Y-%m-%d")}\nQ₀ = {Q0:.2f}',
             xy=(baseline_date, Q0), xytext=(50, 40),
             textcoords='offset points', fontsize=8.5,
             bbox=C_ANNOTBOX,
             arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8))
ax2.set_ylabel('Q (cfs)', fontsize=10)
ax2.set_xlabel('Date', fontsize=10)
ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.9, edgecolor='#cccccc')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax2.tick_params(labelsize=9)

OUT = os.path.join(OUT_DIR, 'schematic_bfd_wte_q.png')
fig.savefig(OUT, dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved → {OUT}")
print(f"Well {WELL_ID}, Gage {GAGE_ID}")
print(f"WTE0={WTE0:.2f} ft, Q0={Q0:.2f} cfs, baseline={baseline_date.date()}")
