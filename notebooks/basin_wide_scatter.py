"""
Basin-wide scatter: ΔQ vs ΔWTE across all gages in the GSLB.

Outputs (results/analysis/scatter/basin_wide/):
  basin_wide_by_gage.png          — raw ΔQ, all obs, colored by gage
  basin_wide_declining_gw.png     — raw ΔQ, ΔWTE < 0 only
  basin_wide_norm_by_gage.png     — normalized ΔQ/Q₀, all obs, colored by gage
  basin_wide_norm_declining_gw.png— normalized ΔQ/Q₀, ΔWTE < 0 only
  basin_wide_facet_raw.png        — 1×5 facet grid, raw ΔQ per gage
  basin_wide_facet_norm.png       — 1×5 facet grid, normalized ΔQ/Q₀ per gage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from scipy.stats import linregress
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "analysis" / "scatter" / "basin_wide"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load & clean ──────────────────────────────────────────────────────────────
data = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv")
data["date"] = pd.to_datetime(data["date"])
data = data.dropna(subset=["delta_wte", "delta_q", "q0"])
data = data[data["q0"] > 0]

# normalized fractional change in flow
data["delta_q_norm"] = data["delta_q"] / data["q0"]

# clip outliers per column (middle 99%)
for col in ["delta_wte", "delta_q", "delta_q_norm"]:
    lo, hi = data[col].quantile([0.005, 0.995])
    data = data[(data[col] >= lo) & (data[col] <= hi)]

gages = sorted(data["gage_id"].unique())
CMAP  = mplcm.get_cmap("tab10")  # noqa: deprecated in 3.7 but still works here
gage_colors = {g: CMAP(i % 10) for i, g in enumerate(gages)}

gage_names = (
    data[["gage_id", "gage_name"]].dropna()
    .drop_duplicates("gage_id")
    .set_index("gage_id")["gage_name"]
    .to_dict()
)

MIN_FIT = 30


def _fit(x, y):
    slope, intercept, r_val, p_val, _ = linregress(x, y)
    p_str = f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.3f}"
    return slope, intercept, r_val**2, p_str


# ── helpers ───────────────────────────────────────────────────────────────────
def _pooled_scatter(df, y_col, y_label, title_suffix, filename):
    """Single basin-wide scatter, colored by gage, with pooled + per-gage fits."""
    fig, ax = plt.subplots(figsize=(11, 7))

    for gage_id in gages:
        sub = df[df["gage_id"] == gage_id]
        if sub.empty:
            continue
        label = f"{gage_id}  ({gage_names.get(gage_id, '')})"
        ax.scatter(sub["delta_wte"], sub[y_col],
                   s=8, color=gage_colors[gage_id], alpha=0.35,
                   edgecolors="none", label=label, zorder=3)

    all_x = df["delta_wte"].values
    all_y = df[y_col].values
    xpad  = (all_x.max() - all_x.min()) * 0.03 or 1
    xlim  = (all_x.min() - xpad, all_x.max() + xpad)
    x_fit = np.linspace(xlim[0], xlim[1], 300)

    for gage_id in gages:
        sub = df[df["gage_id"] == gage_id]
        if len(sub) < MIN_FIT or sub["delta_wte"].std() == 0:
            continue
        slope, intercept, _, _ = _fit(sub["delta_wte"].values, sub[y_col].values)
        ax.plot(x_fit, slope * x_fit + intercept,
                color=gage_colors[gage_id], linewidth=1.3,
                linestyle="--", alpha=0.75, zorder=4)

    if len(all_x) >= MIN_FIT and all_x.std() > 0:
        slope, intercept, r2, p_str = _fit(all_x, all_y)
        ax.plot(x_fit, slope * x_fit + intercept,
                color="black", linewidth=2.2, zorder=6, label="Overall fit")
        unit = "ft⁻¹" if y_col == "delta_q_norm" else "cfs/ft"
        ax.text(0.98, 0.97,
                f"Pooled (all gages)\nN = {len(all_x):,}\n"
                f"Slope = {slope:.4f} {unit}\nR² = {r2:.4f}\np = {p_str}",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.9, edgecolor="#999999"))

    ax.axhline(0, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=2)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=2)
    ax.set_xlabel("ΔWTE (ft)  [+ = rising,  − = declining]", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        f"Basin-wide {y_col.replace('_',' ')} vs ΔWTE — GSLB  {title_suffix}\n"
        f"(dashed = per-gage fit,  solid black = pooled fit)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.88, markerscale=2.5)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = OUT_DIR / filename
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def _facet(df, y_col, y_label, title_suffix, filename):
    """1×5 facet grid — one subplot per gage."""
    n = len(gages)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.5), sharey=False)

    for ax, gage_id in zip(axes, gages):
        sub = df[df["gage_id"] == gage_id].dropna(subset=["delta_wte", y_col])
        color = gage_colors[gage_id]
        name  = gage_names.get(gage_id, str(gage_id))

        ax.scatter(sub["delta_wte"], sub[y_col],
                   s=6, color=color, alpha=0.40, edgecolors="none", zorder=3)

        x = sub["delta_wte"].values
        y = sub[y_col].values
        stats_lines = [f"N = {len(x):,}"]

        if len(x) >= MIN_FIT and x.std() > 0:
            slope, intercept, r2, p_str = _fit(x, y)
            xpad  = (x.max() - x.min()) * 0.05 or 1
            x_fit = np.linspace(x.min() - xpad, x.max() + xpad, 200)
            ax.plot(x_fit, slope * x_fit + intercept,
                    color="red", linewidth=1.6, zorder=5)
            unit = "ft⁻¹" if y_col == "delta_q_norm" else "cfs/ft"
            stats_lines += [f"Slope = {slope:.4f} {unit}",
                            f"R² = {r2:.4f}", f"p = {p_str}"]

        ax.text(0.97, 0.97, "\n".join(stats_lines),
                transform=ax.transAxes, fontsize=7.5,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor="#AAAAAA"))

        ax.axhline(0, color="#CCCCCC", linewidth=0.6, linestyle=":", zorder=2)
        ax.axvline(0, color="#CCCCCC", linewidth=0.6, linestyle=":", zorder=2)
        ax.set_xlabel("ΔWTE (ft)", fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(f"{gage_id}\n{name}", fontsize=8, fontweight="bold")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"ΔQ vs ΔWTE by gage — GSLB  {title_suffix}",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out = OUT_DIR / filename
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── run all six plots ─────────────────────────────────────────────────────────
declining = data[data["delta_wte"] < 0]

# pooled scatter — raw ΔQ
_pooled_scatter(data,     "delta_q",      "ΔQ (cfs)",
                "(all observations)",           "basin_wide_by_gage.png")
_pooled_scatter(declining, "delta_q",     "ΔQ (cfs)",
                "(ΔWTE < 0 — declining GW only)", "basin_wide_declining_gw.png")

# pooled scatter — normalized ΔQ/Q₀
_pooled_scatter(data,     "delta_q_norm", "ΔQ / Q₀  (fractional change in flow)",
                "(all observations — normalized)", "basin_wide_norm_by_gage.png")
_pooled_scatter(declining, "delta_q_norm","ΔQ / Q₀  (fractional change in flow)",
                "(ΔWTE < 0 — normalized)",        "basin_wide_norm_declining_gw.png")

# facet grids
_facet(data,      "delta_q",      "ΔQ (cfs)",
       "(all obs — raw)",          "basin_wide_facet_raw.png")
_facet(data,      "delta_q_norm", "ΔQ / Q₀",
       "(all obs — normalized)",   "basin_wide_facet_norm.png")

print(f"\nAll figures → {OUT_DIR}")
