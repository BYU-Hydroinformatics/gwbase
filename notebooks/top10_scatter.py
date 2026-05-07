"""
For each terminal gage with R²/MI data:
  - Filter data_with_deltas to top-10 wells (by R² or MI)
  - Plot one scatter (ΔWTE vs ΔQ) per gage, colored by well
  - Overlay overall regression line + stats box
Output:
  results/figures/top10_wells_scatter/r2/  (one PNG per gage)
  results/figures/top10_wells_scatter/mi/  (one PNG per gage)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT_R2  = RESULTS / "figures" / "top10_wells_scatter" / "r2"
OUT_MI  = RESULTS / "figures" / "top10_wells_scatter" / "mi"
OUT_R2.mkdir(parents=True, exist_ok=True)
OUT_MI.mkdir(parents=True, exist_ok=True)

data  = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv")
data["date"] = pd.to_datetime(data["date"])
r2_df = pd.read_csv(RESULTS / "features" / "regression_by_well.csv")
mi_df = pd.read_csv(RESULTS / "features" / "mi_analysis.csv")
mi_df = mi_df[mi_df["mi"].notna()]

gage_names = {}
for _, row in r2_df[["gage_id","gage_name"]].dropna().drop_duplicates("gage_id").iterrows():
    gage_names[row["gage_id"]] = row["gage_name"]
for _, row in mi_df[["gage_id","gage_name"]].dropna().drop_duplicates("gage_id").iterrows():
    gage_names.setdefault(row["gage_id"], row["gage_name"])

TOP_N  = 10
CMAP   = cm.get_cmap("tab10")


def plot_gage_scatter(gage_id, top_well_ids, metric_label, metric_vals, out_dir):
    gage_name = gage_names.get(gage_id, "")
    gage_data = data[data["gage_id"] == gage_id]

    sub = gage_data[gage_data["well_id"].astype(str).isin(
        [str(w) for w in top_well_ids]
    )].dropna(subset=["delta_wte", "delta_q"])

    if len(sub) < 2:
        print(f"  Gage {gage_id}: insufficient data – skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot each well with its own color
    unique_wells = sub["well_id"].unique()
    for i, wid in enumerate(unique_wells):
        wdata = sub[sub["well_id"] == wid]
        ax.scatter(wdata["delta_wte"], wdata["delta_q"],
                   s=22, color=CMAP(i % 10), alpha=0.7,
                   edgecolors="none", zorder=3)

    # Overall regression on all points
    x_all = sub["delta_wte"].values
    y_all = sub["delta_q"].values
    slope, intercept, r_val, p_val, _ = linregress(x_all, y_all)
    x_fit = np.linspace(x_all.min(), x_all.max(), 200)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="black", linewidth=1.8, linestyle="--",
            label="Overall regression", zorder=5)

    ax.axhline(0, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=2)
    ax.axvline(0, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=2)

    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    stats_text = (
        f"Wells: {len(unique_wells)}\n"
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

    title = f"Gage {gage_id}  —  {gage_name}\nΔQ vs ΔWTE  (top {len(unique_wells)} wells by {metric_label})"
    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = out_dir / f"{gage_id}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── R² top 10 ─────────────────────────────────────────────────────────────────
print("=== Top 10 by R² ===")
for gage_id, grp in r2_df.groupby("gage_id"):
    print(f"\nGage {gage_id}:")
    ranked = grp.nlargest(TOP_N, "r_squared")
    top_ids = ranked["well_id"].tolist()
    metric_vals = dict(zip(ranked["well_id"].astype(str), ranked["r_squared"]))
    plot_gage_scatter(gage_id, top_ids, "R²", metric_vals, OUT_R2)

# ── MI top 10 ─────────────────────────────────────────────────────────────────
print("\n=== Top 10 by MI ===")
for gage_id, grp in mi_df.groupby("gage_id"):
    print(f"\nGage {gage_id}:")
    ranked = grp.nlargest(TOP_N, "mi")
    top_ids = ranked["well_id"].tolist()
    metric_vals = dict(zip(ranked["well_id"].astype(str), ranked["mi"]))
    plot_gage_scatter(gage_id, top_ids, "MI", metric_vals, OUT_MI)

print(f"\nDone.\n  R² plots → {OUT_R2}\n  MI plots  → {OUT_MI}")
