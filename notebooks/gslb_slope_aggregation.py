"""
GSLB slope aggregation: two basin-scale methods.

Method 1 — Simple Sum
    Sum per-gage OLS slopes → GSLB slope (cfs/ft).
    Plot: single basin-wide scatter (all gages, colored) with per-gage dashed
    fits and the summed-slope line through the data centroid.

Method 2 — Normalized pooled regression
    Normalize each observation by its reference flow (ΔQ / Q₀), pool all
    terminal gages, fit one OLS line → basin-scale sensitivity (ft⁻¹).
    Plot: single scatter colored by gage + pooled fit line.

Outputs → results/analysis/scatter/gslb_aggregation/
    gslb_simple_sum.png
    gslb_normalized.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from scipy.stats import linregress
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "analysis" / "scatter" / "gslb_aggregation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load & clean ──────────────────────────────────────────────────────────────
data = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv")
data["date"] = pd.to_datetime(data["date"])
data = data.dropna(subset=["delta_wte", "delta_q", "q0"])
data = data[data["q0"] > 0]
data["delta_q_norm"] = data["delta_q"] / data["q0"]

# Clip outliers (middle 99%) per column
for col in ["delta_wte", "delta_q", "delta_q_norm"]:
    lo, hi = data[col].quantile([0.005, 0.995])
    data = data[(data[col] >= lo) & (data[col] <= hi)]

gages = sorted(data["gage_id"].unique())
gage_names = (
    data[["gage_id", "gage_name"]].dropna()
    .drop_duplicates("gage_id")
    .set_index("gage_id")["gage_name"]
    .to_dict()
)

# Short labels for plot titles
SHORT = {
    10126000: "Bear River",
    10141000: "Weber River",
    10152000: "Spanish Fork",
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
# Method 1 — Simple Sum
# ══════════════════════════════════════════════════════════════════════════════
def plot_simple_sum():
    # Compute per-gage slopes first
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

    # Scatter all gages, colored by gage
    for gage_id in gages:
        sub   = data[data["gage_id"] == gage_id]
        color = gage_colors[gage_id]
        name  = SHORT.get(gage_id, str(gage_id))
        ax.scatter(sub["delta_wte"], sub["delta_q"],
                   s=7, color=color, alpha=0.35, edgecolors="none",
                   label=f"{name}  ({gage_id})", zorder=3)

    # Per-gage dashed fit lines
    for gage_id, slope in slopes.items():
        sub  = data[data["gage_id"] == gage_id]
        x    = sub["delta_wte"].values
        xpad = (x.max() - x.min()) * 0.03 or 1
        x_fit = np.linspace(x.min() - xpad, x.max() + xpad, 200)
        ax.plot(x_fit, slope * x_fit + intercepts[gage_id],
                color=gage_colors[gage_id], linewidth=1.2,
                linestyle="--", alpha=0.7, zorder=4)

    # Basin summed-slope line through the origin (ΔWTE=0 → ΔQ=0 by construction)
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
    ax.set_title(
        "Simple-Sum Basin Slope  |  GSLB Terminal Gages\n"
        r"(dashed = per-gage fit,  solid black = $\Sigma$ slopes through origin)",
        fontsize=11, fontweight="bold",
    )
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
# Method 2 — Normalized pooled regression
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

    # Per-gage dashed fits
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

    # Pooled basin-scale fit on normalized data
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
    ax.set_title(
        "Normalized Basin-Scale Relationship  |  GSLB Terminal Gages\n"
        "(dashed = per-gage fit,  solid black = pooled regression on ΔQ/Q₀)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.88, markerscale=2.2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = OUT_DIR / "gslb_normalized.png"
    plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  Basin normalized slope = {slope:.6f} ft⁻¹  (R²={r2:.4f}, p={p_str})")
    return slope, r2, p_str


# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Method 1: Simple Sum")
print("=" * 60)
basin_slope = plot_simple_sum()

print()
print("=" * 60)
print("Method 2: Normalized pooled regression")
print("=" * 60)
norm_slope, norm_r2, norm_p = plot_normalized()

print()
print("── Summary ──────────────────────────────────────────────────")
print(f"  Simple sum slope   : {basin_slope:.4f}  cfs/ft")
print(f"  Normalized slope   : {norm_slope:.6f}  ft⁻¹  (R²={norm_r2:.4f}, p={norm_p})")
print(f"  Output dir         : {OUT_DIR}")
