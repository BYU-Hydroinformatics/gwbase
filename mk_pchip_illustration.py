#!/usr/bin/env python3
"""
Generate a publication-quality figure illustrating PCHIP monthly interpolation.

Shows original sparse observations alongside PCHIP-interpolated monthly values
for a representative groundwater well.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import PchipInterpolator
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
WELL_ID = 403859111535601   # Representative well (WTE ~4293 ft, 1990–1992)
DATA_PATH = 'data/raw/groundwater/GSLB_1900-2025_TS_with_aquifers.csv'
OUTPUT_PATH = 'results/figures/pchip_monthly_illustration.pdf'

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

obs = df[df['Well_ID'] == WELL_ID].sort_values('Date').reset_index(drop=True)
obs = obs[['Date', 'WTE']].rename(columns={'Date': 'date', 'WTE': 'wte'})
print(f"Loaded {len(obs)} observations for well {WELL_ID}")
print(obs.to_string(index=False))

# ── PCHIP interpolation to monthly resolution ─────────────────────────────────
x_obs = obs['date'].map(pd.Timestamp.toordinal).values
y_obs = obs['wte'].values

interpolator = PchipInterpolator(x_obs, y_obs)

# Generate monthly dates at the 15th of each month
first_month = obs['date'].min().to_period('M').to_timestamp()
last_month  = obs['date'].max().to_period('M').to_timestamp()
month_starts = pd.date_range(start=first_month, end=last_month, freq='MS')
monthly_dates = month_starts + pd.offsets.Day(14)   # ~mid-month (15th)

x_monthly = monthly_dates.map(pd.Timestamp.toordinal).values
y_monthly  = interpolator(x_monthly)

monthly = pd.DataFrame({'date': monthly_dates, 'wte': y_monthly})
print(f"\nGenerated {len(monthly)} monthly interpolated values")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3.2))

# Monthly interpolated line
ax.plot(monthly['date'], monthly['wte'],
        color='#1a4f9e', linewidth=1.6, zorder=2,
        label='PCHIP interpolated (monthly)')

# Monthly interpolated points (small, filled)
ax.scatter(monthly['date'], monthly['wte'],
           color='#1a4f9e', s=18, zorder=3, linewidths=0)

# Original observations
ax.scatter(obs['date'], obs['wte'],
           color='#d62728', s=40, zorder=4, linewidths=0.5,
           edgecolors='white',
           label='Original observations')

# Axes formatting
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8.5)
ax.yaxis.set_tick_params(labelsize=8.5)

ax.set_xlabel('Date', fontsize=9.5, labelpad=4)
ax.set_ylabel('Water Table Elevation (ft)', fontsize=9.5, labelpad=4)
ax.set_title('PCHIP Interpolation (Monthly)', fontsize=10.5, fontweight='bold', pad=6)

ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#cccccc',
          loc='lower right')

ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, color='#999999')
ax.set_axisbelow(True)

fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\nSaved figure to: {OUTPUT_PATH}")

# Also save as PNG for quick preview
png_path = OUTPUT_PATH.replace('.pdf', '.png')
fig.savefig(png_path, dpi=200, bbox_inches='tight')
print(f"Saved PNG preview to: {png_path}")

plt.show()
