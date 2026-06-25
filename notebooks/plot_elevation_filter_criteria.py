"""
Conceptual diagram: Reach Elevation vs Water Table Elevation filtering criteria.
Buffer distance = 30 m.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'analysis', 'filter_sensitivity')
os.makedirs(OUT_DIR, exist_ok=True)

REACH_ELEV = 100   # m — the horizontal blue line
BUFFER     = 30    # m
Y_MIN, Y_MAX = 40, 140
X_MIN, X_MAX = 0, 900

# Sample points: (x, wte, delta, keep, label_offset_pts, label_va)
# delta = reach - wte; keep if delta <= 30
# label_offset_pts: (x_pts, y_pts); label_va: 'bottom' or 'top'
points = [
    (100, 115, -15, True,   (0,  12), 'bottom'),
    (200,  95,   5, True,   (0, -14), 'top'),    # near reach line → label below
    (300,  85,  15, True,   (0,  12), 'bottom'),
    (400,  75,  25, True,   (0,  12), 'bottom'),
    (500,  90,  10, True,   (0,  12), 'bottom'),
    (590, 110, -10, True,   (0,  12), 'bottom'),
    (700,  50,  50, False,  (0,  12), 'bottom'),
    (800,  88,  12, True,   (0,  12), 'bottom'),
]

fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

# ── Background zones ──────────────────────────────────────────────────────────
# Green: WTE > reach (gaining) → delta < 0, i.e. wte in [REACH_ELEV, Y_MAX]
ax.fill_between([X_MIN, X_MAX], REACH_ELEV, Y_MAX,
                color='#c8e6c9', alpha=0.7, zorder=0)
# Blue: reach - buffer ≤ WTE ≤ reach → wte in [REACH_ELEV - BUFFER, REACH_ELEV]
ax.fill_between([X_MIN, X_MAX], REACH_ELEV - BUFFER, REACH_ELEV,
                color='#bbdefb', alpha=0.7, zorder=0)
# Red: WTE < reach - buffer → wte in [Y_MIN, REACH_ELEV - BUFFER]
ax.fill_between([X_MIN, X_MAX], Y_MIN, REACH_ELEV - BUFFER,
                color='#ffcdd2', alpha=0.7, zorder=0)

# ── Reach elevation line ──────────────────────────────────────────────────────
ax.axhline(REACH_ELEV, color='#1a237e', linewidth=2.5, zorder=3, label='Reach Elevation')

# ── Data points ───────────────────────────────────────────────────────────────
for x, wte, delta, keep, offset, va in points:
    if keep:
        ax.plot(x, wte, 'o', color='#2e7d32', markersize=10, zorder=5)
        ax.annotate(f'Δ={delta:+d}m\nKEEP', xy=(x, wte),
                    xytext=offset, textcoords='offset points',
                    ha='center', va=va, fontsize=8,
                    color='#2e7d32', fontweight='bold')
    else:
        ax.plot(x, wte, 'x', color='#c62828', markersize=12,
                markeredgewidth=2.5, zorder=5)
        ax.annotate(f'Δ={delta:+d}m\nEXCLUDE', xy=(x, wte),
                    xytext=offset, textcoords='offset points',
                    ha='center', va=va, fontsize=8,
                    color='#c62828', fontweight='bold')

# ── Axes labels ───────────────────────────────────────────────────────────────
ax.set_xlabel('Arbitrary horizontal position', fontsize=11)
ax.set_ylabel('Elevation (m)', fontsize=11)
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.tick_params(labelsize=10)

# ── Legend ────────────────────────────────────────────────────────────────────
reach_line  = plt.Line2D([0], [0], color='#1a237e', linewidth=2.5, label='Reach Elevation')
keep_gain   = mpatches.Patch(facecolor='#c8e6c9', alpha=0.8,
                              label='KEPT: WTE > Reach (Gaining stream)')
keep_buf    = mpatches.Patch(facecolor='#bbdefb', alpha=0.8,
                              label='KEPT: WTE < Reach but within buffer distance')
excl_patch  = mpatches.Patch(facecolor='#ffcdd2', alpha=0.8,
                              label='EXCLUDED: (Reach − WTE) > buffer distance')
ax.legend(handles=[reach_line, keep_gain, keep_buf, excl_patch],
          loc='upper right', fontsize=9, framealpha=0.9)

ax.grid(False)

OUT = os.path.join(OUT_DIR, 'elevation_filter_criteria_diagram.png')
fig.savefig(OUT, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {OUT}")
