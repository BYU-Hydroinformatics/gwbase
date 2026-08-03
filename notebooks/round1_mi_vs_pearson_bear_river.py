"""
Round-1 revision -- fix media/figure15.png (fig:mi_vs_pearson).

The manuscript caption and Section 5.3 prose report the MI-vs-|Pearson r|
comparison for the Bear River terminal gage only (r = 0.311, p = 0.0018,
n = 98). The image file actually in media/figure15.png was, instead, the
basin-wide all-five-gage version (r = 0.428, p = 1.36e-13, n = 273) --
matplotlib's default `plot_mi_results` output in gwbase/visualization.py,
which always plots whatever `mi_results` dataframe it is handed and has no
per-gage filtering. A correct Bear-River-only version existed once
(commit 745df62, 2026-06-02) but was silently overwritten by a later
"regenerate figures" pass (commit 67772c6, 2026-06-09) that never touched
this file's caption/text, so the mismatch went unnoticed. The custom
script that made the original 2026-06-02 fix was not preserved in the
repo, so this regenerates it from scratch.

Filters results/features/mi_analysis.csv to gage_id 10126000 (Bear River
near Corinne) before computing the OLS trend -- reproduces the
manuscript's stated numbers exactly (r=0.311, p=0.0018, n=98, 1 of 99
Bear River pairs dropped for NaN).

Run:  ./.venv/bin/python notebooks/round1_mi_vs_pearson_bear_river.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

BASE = Path(__file__).parent.parent
SRC = BASE / "results" / "features" / "mi_analysis.csv"
OUT = BASE / "results" / "round1_revision" / "08_mi_vs_pearson_bear_river"
OUT.mkdir(parents=True, exist_ok=True)

BEAR_RIVER_GAGE_ID = 10126000

df = pd.read_csv(SRC)
bear = df[df["gage_id"] == BEAR_RIVER_GAGE_ID].copy()
print(f"Bear River well-gage pairs: {len(bear)}")

x = bear["pearson_r"].abs()
y = bear["mi"]
mask = ~(x.isna() | y.isna())
print(f"Valid (non-NaN) pairs: {mask.sum()} (dropped {(~mask).sum()})")

xc, yc = x[mask].values, y[mask].values
slope, intercept, r_val, p_val, _ = linregress(xc, yc)
print(f"r={r_val:.3f}  p={p_val:.4f}  n={len(xc)}  slope={slope:.4f}  intercept={intercept:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(xc, yc, alpha=0.5, s=20, color="steelblue", zorder=2)
xs = [xc.min(), xc.max()]
ax.plot(xs, [slope * v + intercept for v in xs], color="red", linewidth=1.8, label="OLS trend")
p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}"
ax.text(
    0.05, 0.95, f"r = {r_val:.3f}\np = {p_str}\nn = {len(xc)}",
    transform=ax.transAxes, va="top", ha="left", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)
ax.set_xlabel("|Pearson r|")
ax.set_ylabel("Mutual Information")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()

png_out = OUT / "mi_vs_pearson_bear_river.png"
fig.savefig(png_out, dpi=600)
print(f"Saved -> {png_out}")

csv_out = OUT / "bear_river_mi_pearson_pairs.csv"
bear[mask].to_csv(csv_out, index=False)
print(f"Saved -> {csv_out}")
