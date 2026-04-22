"""
For each gage, plot the slope distribution of ALL wells (regression_by_well.csv),
marking top-10 R² and top-10 MI wells.

Output:
  results/figures/slope_distribution_by_gage.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT     = RESULTS / "figures" / "slope_distribution_by_gage.png"

# ── Load data ────────────────────────────────────────────────────────────────
reg   = pd.read_csv(RESULTS / "features" / "regression_by_well.csv")
top_r2 = pd.read_csv(RESULTS / "figures" / "top10_wells_scatter" / "tables" / "top10_by_r2.csv")
top_mi = pd.read_csv(RESULTS / "figures" / "top10_wells_scatter" / "tables" / "top10_by_mi.csv")

reg["well_id"]    = reg["well_id"].astype(str)
reg["gage_id"]    = reg["gage_id"].astype(str)
top_r2["well_id"] = top_r2["well_id"].astype(str)
top_r2["gage_id"] = top_r2["gage_id"].astype(str)
top_mi["well_id"] = top_mi["well_id"].astype(str)
top_mi["gage_id"] = top_mi["gage_id"].astype(str)

# Short gage labels
gage_order = [
    "BEAR RIVER NEAR CORINNE - UT",
    "WEBER RIVER NEAR PLAIN CITY - UT",
    "PROVO RIVER AT PROVO - UT",
    "SPANISH FORK NEAR LAKE SHORE - UTAH",
    "LITTLE COTTONWOOD CREEK @ JORDAN RIVER NR SLC",
]
gage_short = {
    "BEAR RIVER NEAR CORINNE - UT":                   "Bear River\n(n=96)",
    "WEBER RIVER NEAR PLAIN CITY - UT":                "Weber River\n(n=101)",
    "PROVO RIVER AT PROVO - UT":                       "Provo River\n(n=46)",
    "SPANISH FORK NEAR LAKE SHORE - UTAH":             "Spanish Fork\n(n=7)",
    "LITTLE COTTONWOOD CREEK @ JORDAN RIVER NR SLC":   "Little Cottonwood\n(n=15)",
}

# ── Figure: 5 rows (one per gage) × 2 cols ───────────────────────────────────
fig, axes = plt.subplots(5, 2, figsize=(14, 18),
                         gridspec_kw={"width_ratios": [2.2, 1]})
fig.suptitle(
    "Slope Distribution — All Wells per Gage\n"
    "Red ticks = top-10 by R²   |   Blue ticks = top-10 by MI",
    fontsize=13, fontweight="bold", y=0.995
)

for row, gage_name in enumerate(gage_order):
    ax_hist = axes[row, 0]
    ax_stat = axes[row, 1]

    sub   = reg[reg["gage_name"] == gage_name].copy()
    n_all = len(sub)
    n_neg = (sub["slope"] < 0).sum()
    n_pos = (sub["slope"] > 0).sum()

    # top-10 R² and MI for this gage
    t_r2 = top_r2[top_r2["gage_name"] == gage_name]["well_id"].tolist()
    t_mi = top_mi[top_mi["gage_name"] == gage_name]["well_id"].tolist()

    slopes_r2_neg = sub[sub["well_id"].isin(t_r2) & (sub["slope"] < 0)]["slope"].values
    slopes_r2_pos = sub[sub["well_id"].isin(t_r2) & (sub["slope"] >= 0)]["slope"].values
    slopes_mi_neg = sub[sub["well_id"].isin(t_mi) & (sub["slope"] < 0)]["slope"].values
    slopes_mi_pos = sub[sub["well_id"].isin(t_mi) & (sub["slope"] >= 0)]["slope"].values

    slopes = sub["slope"].values

    # Clip extreme outliers for display only (keep stats on full data)
    p1, p99 = np.percentile(slopes, 1), np.percentile(slopes, 99)
    clip_lo = max(slopes.min(), p1 - abs(p1)*0.1)
    clip_hi = min(slopes.max(), p99 + abs(p99)*0.1)
    slopes_clipped = np.clip(slopes, clip_lo, clip_hi)

    # ── Histogram ──────────────────────────────────────────────────────────
    n_bins = min(40, max(15, n_all // 3))
    counts, edges, patches = ax_hist.hist(
        slopes_clipped, bins=n_bins, color="#AAAAAA", alpha=0.6,
        edgecolor="white", linewidth=0.4, label="All wells"
    )
    # Colour negative bars
    for patch, left in zip(patches, edges[:-1]):
        if left < 0:
            patch.set_facecolor("#E24A33")
            patch.set_alpha(0.55)
        else:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.55)

    # Vertical line at 0
    ax_hist.axvline(0, color="black", linewidth=1.2, linestyle="--", zorder=5)

    # Tick marks for top-10 wells (bottom of plot)
    ymax = counts.max()
    tick_y_r2 = -ymax * 0.08
    tick_y_mi = -ymax * 0.16
    ax_hist.set_ylim(tick_y_mi * 1.5, ymax * 1.25)

    def draw_ticks(ax, xs, y, color, marker, size=80):
        xs_c = np.clip(xs, clip_lo, clip_hi)
        ax.scatter(xs_c, np.full(len(xs_c), y), marker=marker,
                   color=color, s=size, zorder=6, clip_on=False)

    draw_ticks(ax_hist, slopes_r2_neg, tick_y_r2, "#E24A33", "v", 60)
    draw_ticks(ax_hist, slopes_r2_pos, tick_y_r2, "#4C72B0", "^", 60)
    draw_ticks(ax_hist, slopes_mi_neg, tick_y_mi, "#E24A33", "D", 40)
    draw_ticks(ax_hist, slopes_mi_pos, tick_y_mi, "#4C72B0", "D", 40)

    # Labels
    ax_hist.set_ylabel("Count", fontsize=9)
    if row == 4:
        ax_hist.set_xlabel("Slope  (ΔQ / ΔWTE,  cfs / ft)", fontsize=9)
    ax_hist.set_title(
        f"{gage_short[gage_name]}   "
        f"Neg: {n_neg}/{n_all} ({100*n_neg/n_all:.0f}%)   "
        f"Pos: {n_pos}/{n_all} ({100*n_pos/n_all:.0f}%)",
        fontsize=9, fontweight="bold", loc="left"
    )
    ax_hist.grid(axis="y", alpha=0.25)

    # Tick legend (first row only)
    if row == 0:
        from matplotlib.lines import Line2D
        legend_els = [
            Line2D([0],[0], marker="v", color="#E24A33", linestyle="None",
                   markersize=7, label="Top-10 R² (neg slope)"),
            Line2D([0],[0], marker="^", color="#4C72B0", linestyle="None",
                   markersize=7, label="Top-10 R² (pos slope)"),
            Line2D([0],[0], marker="D", color="#E24A33", linestyle="None",
                   markersize=6, label="Top-10 MI  (neg slope)"),
            Line2D([0],[0], marker="D", color="#4C72B0", linestyle="None",
                   markersize=6, label="Top-10 MI  (pos slope)"),
        ]
        ax_hist.legend(handles=legend_els, fontsize=7.5, loc="upper right",
                       framealpha=0.85)

    # ── Stats panel ────────────────────────────────────────────────────────
    ax_stat.axis("off")

    # R²-sorted top-10 slopes
    top_r2_sub = sub[sub["well_id"].isin(t_r2)].sort_values("r_squared", ascending=False)
    top_mi_sub = sub[sub["well_id"].isin(t_mi)].sort_values("r_squared", ascending=False)

    lines = [
        f"{'ALL WELLS':^32}",
        f"  median slope : {sub['slope'].median():+.2f}",
        f"  mean slope   : {sub['slope'].mean():+.2f}",
        f"  negative     : {n_neg}/{n_all} ({100*n_neg/n_all:.0f}%)",
        "",
        f"{'TOP-10 R²  (slope)':^32}",
    ]
    for _, r in top_r2_sub.iterrows():
        sign_char = "▼" if r["slope"] < 0 else "▲"
        lines.append(f"  {sign_char} {r['slope']:+7.2f}   R²={r['r_squared']:.3f}")

    lines += ["", f"{'TOP-10 MI  (slope via reg)':^32}"]
    for _, r in top_mi_sub.iterrows():
        sign_char = "▼" if r["slope"] < 0 else "▲"
        lines.append(f"  {sign_char} {r['slope']:+7.2f}   R²={r['r_squared']:.3f}")

    ax_stat.text(0.02, 0.97, "\n".join(lines),
                 transform=ax_stat.transAxes,
                 fontsize=7.2, va="top", ha="left", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F8F8",
                           edgecolor="#CCCCCC", alpha=0.95))

plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUT}")
