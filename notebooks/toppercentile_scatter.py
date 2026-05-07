"""
For each terminal gage, select wells in the top 10% and top 25% by R²
(from all wells that have R² data), then plot ΔQ vs ΔWTE scatter plots.

Output:
  results/figures/top_percentile_scatter/top10pct/  (one PNG per gage)
  results/figures/top_percentile_scatter/top25pct/  (one PNG per gage)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT_10  = RESULTS / "figures" / "top_percentile_scatter" / "top10pct"
OUT_25  = RESULTS / "figures" / "top_percentile_scatter" / "top25pct"
OUT_10.mkdir(parents=True, exist_ok=True)
OUT_25.mkdir(parents=True, exist_ok=True)

data  = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv")
data["date"] = pd.to_datetime(data["date"])
r2_df = pd.read_csv(RESULTS / "features" / "regression_by_well.csv")

gage_names = (
    r2_df[["gage_id", "gage_name"]].dropna()
    .drop_duplicates("gage_id")
    .set_index("gage_id")["gage_name"]
    .to_dict()
)

CMAP = cm.tab20


def plot_percentile_scatter(gage_id, top_well_ids, pct_label, n_total, out_dir):
    gage_name = gage_names.get(gage_id, "")
    gage_data = data[data["gage_id"] == gage_id]

    sub = gage_data[gage_data["well_id"].astype(str).isin(
        [str(w) for w in top_well_ids]
    )].dropna(subset=["delta_wte", "delta_q"])

    if len(sub) < 2:
        print(f"  Gage {gage_id}: insufficient data – skipping.")
        return

    unique_wells = sub["well_id"].unique()
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, wid in enumerate(unique_wells):
        wdata = sub[sub["well_id"] == wid]
        ax.scatter(wdata["delta_wte"], wdata["delta_q"],
                   s=22, color=CMAP(i % 20), alpha=0.7,
                   edgecolors="none", zorder=3)

    x_all = sub["delta_wte"].values
    y_all = sub["delta_q"].values
    slope, intercept, r_val, p_val, _ = linregress(x_all, y_all)
    x_fit = np.linspace(x_all.min(), x_all.max(), 200)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="black", linewidth=1.8, linestyle="--", zorder=5)

    ax.axhline(0, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=2)
    ax.axvline(0, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=2)

    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    stats_text = (
        f"Wells: {len(unique_wells)} / {n_total} (top {pct_label})\n"
        f"N: {len(sub)}\n"
        f"Slope: {slope:.3f}\n"
        f"R²: {r_val**2:.3f}\n"
        f"p: {p_str}"
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#AAAAAA"))

    ax.set_xlabel("ΔWTE (ft)", fontsize=11)
    ax.set_ylabel("ΔQ (cfs)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"Gage {gage_id}  —  {gage_name}\n"
        f"ΔQ vs ΔWTE  (top {pct_label} wells by R²)",
        fontsize=11, fontweight="bold"
    )

    plt.tight_layout()
    out_path = out_dir / f"{gage_id}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


for gage_id, grp in r2_df.groupby("gage_id"):
    print(f"\nGage {gage_id}  ({len(grp)} wells with R²):")

    n_total = len(grp)
    threshold_10 = grp["r_squared"].quantile(0.90)
    threshold_25 = grp["r_squared"].quantile(0.75)

    top10_ids = grp[grp["r_squared"] >= threshold_10]["well_id"].tolist()
    top25_ids = grp[grp["r_squared"] >= threshold_25]["well_id"].tolist()

    print(f"  top 10%: {len(top10_ids)} wells  (R² ≥ {threshold_10:.3f})")
    print(f"  top 25%: {len(top25_ids)} wells  (R² ≥ {threshold_25:.3f})")

    plot_percentile_scatter(gage_id, top10_ids, "10%", n_total, OUT_10)
    plot_percentile_scatter(gage_id, top25_ids, "25%", n_total, OUT_25)

print(f"\nDone.\n  top 10% → {OUT_10}\n  top 25% → {OUT_25}")
