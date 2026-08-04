"""
Round-1 revision -- regenerate manuscript Figures 17/18 with Spanish Fork
(gage 10152000) dropped entirely, not just excluded from the summary numbers.

Supersedes the "shown for reference" behavior in notebooks/gslb_slope_aggregation.py
(commit 864aa8c, 2026-08-02). Advisor review of the round-1 response caught that the
letter to Reviewer 1 says "we now report Spanish Fork once, in Section 5 and in the
Limitations" -- but Spanish Fork's points and dashed fit line were still drawn on
these two basin-wide figures. Reviewer 1's original comment specifically objected to
Spanish Fork "featuring prominently in rankings and figures alongside data-rich
regions," so advisor decision: drop it from the plot entirely, not just the fit.

Same source data and computation as gslb_slope_aggregation.py, just filtered to the
four retained catchments (Bear River, Weber River, Provo River, Little Cottonwood)
before any plotting happens, so Spanish Fork never appears -- not in the scatter, not
in the legend, not in the dashed per-gage fit lines.

Run:  ./.venv/bin/python notebooks/round1_gslb_figures_no_spanish_fork.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from scipy.stats import linregress
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "round1_revision" / "09_gslb_figures_no_spanish_fork"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPANISH_FORK = 10152000

# ── Load & clean ──────────────────────────────────────────────────────────────
data = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv")
data["date"] = pd.to_datetime(data["date"])
data = data.dropna(subset=["delta_wte", "delta_q", "q0"])
data = data[data["q0"] > 0]
data["delta_q_norm"] = data["delta_q"] / data["q0"]

# Drop Spanish Fork entirely, before clipping/plotting -- it never appears below.
data = data[data["gage_id"] != SPANISH_FORK]

# Clip outliers (middle 99%) per column
for col in ["delta_wte", "delta_q", "delta_q_norm"]:
    lo, hi = data[col].quantile([0.005, 0.995])
    data = data[(data[col] >= lo) & (data[col] <= hi)]

gages = sorted(data["gage_id"].unique())
print(f"Retained gages (Spanish Fork dropped): {gages}")

SHORT = {
    10126000: "Bear River",
    10141000: "Weber River",
    10163000: "Provo River",
    10168000: "Little Cottonwood",
}

CMAP = mplcm.get_cmap("tab10")
gage_colors = {g: CMAP(i % 10) for i, g in enumerate(gages)}
MIN_FIT = 30


def _fit(x, y):
    slope, intercept, r_val, p_val, _ = linregress(x, y)
    p_str = f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.3f}"
    return slope, intercept, r_val ** 2, p_str


# ══════════════════════════════════════════════════════════════════════════════
# Method 1 — Simple Sum (manuscript Figure 17, fig:gslb_simple_sum)
# ══════════════════════════════════════════════════════════════════════════════
def plot_simple_sum():
    slopes = {}
    intercepts = {}
    for gage_id in gages:
        sub = data[data["gage_id"] == gage_id]
        x, y = sub["delta_wte"].values, sub["delta_q"].values
        if len(x) >= MIN_FIT and x.std() > 0:
            slope, intercept, _, _ = _fit(x, y)
            slopes[gage_id] = slope
            intercepts[gage_id] = intercept

    basin_total = sum(slopes.values())

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for gage_id in gages:
        sub   = data[data["gage_id"] == gage_id]
        color = gage_colors[gage_id]
        name  = SHORT.get(gage_id, str(gage_id))
        ax.scatter(sub["delta_wte"], sub["delta_q"],
                   s=7, color=color, alpha=0.35, edgecolors="none",
                   label=f"{name}  ({gage_id})", zorder=3)

    for gage_id, slope in slopes.items():
        sub  = data[data["gage_id"] == gage_id]
        x    = sub["delta_wte"].values
        xpad = (x.max() - x.min()) * 0.03 or 1
        x_fit = np.linspace(x.min() - xpad, x.max() + xpad, 200)
        ax.plot(x_fit, slope * x_fit + intercepts[gage_id],
                color=gage_colors[gage_id], linewidth=1.2,
                linestyle="--", alpha=0.7, zorder=4)

    all_x = data["delta_wte"].values
    xpad  = (all_x.max() - all_x.min()) * 0.03 or 1
    x_fit = np.linspace(all_x.min() - xpad, all_x.max() + xpad, 300)
    ax.plot(x_fit, basin_total * x_fit,
            color="black", linewidth=2.2, zorder=6, label="Basin summed slope")

    ax.text(0.98, 0.97,
            f"GSLB simple sum\n"
            f"N gages = {len(slopes)}\n"
            f"Slope = {basin_total:.4f} cfs/ft\n"
            f"(= Σ per-gage slopes)",
            transform=ax.transAxes, fontsize=9.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.92, edgecolor="#999999"))

    ax.axhline(0, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=2)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=2)
    ax.set_xlabel("ΔWTE (ft)  [+ = rising,  − = declining]", fontsize=12)
    ax.set_ylabel("ΔQ (cfs)", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.88, markerscale=2.2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = OUT_DIR / "gslb_simple_sum.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  Basin simple-sum slope = {basin_total:.4f} cfs/ft")
    return basin_total


# ══════════════════════════════════════════════════════════════════════════════
# Method 2 — Normalized pooled regression (manuscript Figure 18, fig:gslb_normalized)
# ══════════════════════════════════════════════════════════════════════════════
def plot_normalized():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    for gage_id in gages:
        sub   = data[data["gage_id"] == gage_id]
        color = gage_colors[gage_id]
        name  = SHORT.get(gage_id, str(gage_id))
        ax.scatter(sub["delta_wte"], sub["delta_q_norm"],
                   s=7, color=color, alpha=0.35, edgecolors="none",
                   label=f"{name}  ({gage_id})", zorder=3)

    for gage_id in gages:
        sub = data[data["gage_id"] == gage_id]
        x, y = sub["delta_wte"].values, sub["delta_q_norm"].values
        if len(x) >= MIN_FIT and x.std() > 0:
            slope, intercept, _, _ = _fit(x, y)
            xpad  = (x.max() - x.min()) * 0.03 or 1
            x_fit = np.linspace(x.min() - xpad, x.max() + xpad, 200)
            ax.plot(x_fit, slope * x_fit + intercept,
                    color=gage_colors[gage_id], linewidth=1.2,
                    linestyle="--", alpha=0.7, zorder=4)

    all_x = data["delta_wte"].values
    all_y = data["delta_q_norm"].values
    xpad  = (all_x.max() - all_x.min()) * 0.03 or 1
    xlim  = (all_x.min() - xpad, all_x.max() + xpad)
    x_fit = np.linspace(xlim[0], xlim[1], 300)

    slope, intercept, r2, p_str = _fit(all_x, all_y)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="black", linewidth=2.2, zorder=6, label="Basin pooled fit")

    ax.text(0.98, 0.97,
            f"GSLB pooled (normalized)\n"
            f"N = {len(all_x):,}\n"
            f"Slope = {slope:.5f} ft⁻¹\n"
            f"R² = {r2:.4f}\n"
            f"p = {p_str}",
            transform=ax.transAxes, fontsize=9.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.92, edgecolor="#999999"))

    ax.axhline(0, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=2)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=2)
    ax.set_xlabel("ΔWTE (ft)  [+ = rising,  − = declining]", fontsize=12)
    ax.set_ylabel("ΔQ / Q₀  (fractional change in streamflow)", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.88, markerscale=2.2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = OUT_DIR / "gslb_normalized.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  Basin normalized slope = {slope:.6f} ft⁻¹  (R²={r2:.4f}, p={p_str})")
    return slope, r2, p_str


print("=" * 60)
print("Method 1: Simple Sum (Spanish Fork dropped entirely)")
print("=" * 60)
basin_slope = plot_simple_sum()

print()
print("=" * 60)
print("Method 2: Normalized pooled regression (Spanish Fork dropped entirely)")
print("=" * 60)
norm_slope, norm_r2, norm_p = plot_normalized()

print()
print("── Summary ──────────────────────────────────────────────────")
print(f"  Simple sum slope   : {basin_slope:.4f}  cfs/ft")
print(f"  Normalized slope   : {norm_slope:.6f}  ft⁻¹  (R²={norm_r2:.4f}, p={norm_p})")
print(f"  Output dir         : {OUT_DIR}")
