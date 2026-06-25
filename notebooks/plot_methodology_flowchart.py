"""
9-step methodology flowchart — recreated from original PPT at 600 DPI.
Snake layout: col 1 down (Steps 1–3), col 2 up (Steps 4–6), col 3 down (Steps 7–9).
"""

import os
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures',
                   'methodology_flowchart.png')

# ── Layout (1 data unit = 1 inch) ────────────────────────────────────────────
# Figure 13 × 9 in; boxes 3.7 × 2.0 in; gaps 0.65 in (H) / 0.75 in (V)
# Side margins ≈ 0.3 in; top/bottom margins ≈ 0.75 in
FIG_W, FIG_H = 13, 9

COL_X = [2.15, 6.50, 10.85]    # col centres
ROW_Y = [7.25, 4.50,  1.75]    # row centres (top → bottom)
BOX_W, BOX_H = 3.70, 2.00
PAD = 0.10                       # FancyBboxPatch exterior pad

# ── Style ─────────────────────────────────────────────────────────────────────
BOX_FC   = '#BDD7EE'
BOX_EC   = '#9DC3E6'
ARROW_C  = '#111111'
FS       = 11.0
LS       = 1.50    # line spacing

# ── Step labels ───────────────────────────────────────────────────────────────
# Line breaks chosen so no line stands alone and lengths are balanced.
STEPS = {
    (0, 0): "Step 1: Identify\nstream network and\nupstream catchments",
    (0, 1): "Step 2: Locate\nGroundwater Wells\nwithin Catchments",
    (0, 2): "Step 3: Associate Wells\nwith Nearest\nStream Segments",
    (1, 2): "Step 4: Filter Wells\nwith Insufficient Data",
    (1, 1): "Step 5: Temporal\nInterpolation of\nGroundwater Levels",
    (1, 0): "Step 6: Elevation-Based\nFiltering",
    (2, 0): "Step 7: Pair Groundwater\nand Streamflow Records\nunder Baseflow-\nDominated Conditions",
    (2, 1): "Step 8: Compute\nΔWTE and ΔQ",
    (2, 2): "Step 9: Analyze\nΔWTE–ΔQ Relationships",
}

ARROWS = [
    ((0, 0), (0, 1)), ((0, 1), (0, 2)),   # 1→2→3  (down col 1)
    ((0, 2), (1, 2)),                       # 3→4    (right)
    ((1, 2), (1, 1)), ((1, 1), (1, 0)),   # 4→5→6  (up col 2)
    ((1, 0), (2, 0)),                       # 6→7    (right)
    ((2, 0), (2, 1)), ((2, 1), (2, 2)),   # 7→8→9  (down col 3)
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1], facecolor='white')
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')

# ── Boxes ─────────────────────────────────────────────────────────────────────
# Shrink spec by PAD on each side so the drawn box (spec + PAD) equals BOX_W × BOX_H.
for (col, row), label in STEPS.items():
    cx, cy = COL_X[col], ROW_Y[row]
    patch = FancyBboxPatch(
        (cx - BOX_W / 2 + PAD,  cy - BOX_H / 2 + PAD),
        BOX_W - 2 * PAD,        BOX_H - 2 * PAD,
        boxstyle=f'round,pad={PAD}',
        facecolor=BOX_FC, edgecolor=BOX_EC, linewidth=0.9, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, label,
            ha='center', va='center', zorder=3,
            fontsize=FS, fontfamily='sans-serif',
            fontweight='bold', multialignment='center', linespacing=LS)

# ── Arrows ────────────────────────────────────────────────────────────────────
AP = dict(arrowstyle='-|>', color=ARROW_C, lw=2.8,
          mutation_scale=24, shrinkA=0, shrinkB=0)

HW, HH = BOX_W / 2, BOX_H / 2

for (fc, fr), (tc, tr) in ARROWS:
    cx0, cy0 = COL_X[fc], ROW_Y[fr]
    cx1, cy1 = COL_X[tc], ROW_Y[tr]

    if fc == tc:              # vertical
        if tr > fr:           # downward
            x0, y0, x1, y1 = cx0, cy0 - HH, cx1, cy1 + HH
        else:                 # upward
            x0, y0, x1, y1 = cx0, cy0 + HH, cx1, cy1 - HH
    else:                     # horizontal
        if tc > fc:           # rightward
            x0, y0, x1, y1 = cx0 + HW, cy0, cx1 - HW, cy1
        else:
            x0, y0, x1, y1 = cx0 - HW, cy0, cx1 + HW, cy1

    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=AP)

fig.savefig(OUT, dpi=600, bbox_inches='tight', pad_inches=0.20, facecolor='white')
plt.close(fig)
print(f"Saved → {OUT}")
