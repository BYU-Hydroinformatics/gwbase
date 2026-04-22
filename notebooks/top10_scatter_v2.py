"""
Extended top-10 well scatter plots (v2).

New outputs (existing results untouched):
  results/figures/top10_wells_scatter/r2_per_well/   — individual subplot per well
  results/figures/top10_wells_scatter/r2_per_well_fit/ — original scatter + per-well fit lines
  results/figures/top10_wells_scatter/mi_per_well/
  results/figures/top10_wells_scatter/mi_per_well_fit/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
from pathlib import Path
import math

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"

# Output dirs (new, preserving originals)
DIRS = {
    "r2": {
        "per_well":     RESULTS / "figures" / "top10_wells_scatter" / "r2_per_well",
        "per_well_fit": RESULTS / "figures" / "top10_wells_scatter" / "r2_per_well_fit",
    },
    "mi": {
        "per_well":     RESULTS / "figures" / "top10_wells_scatter" / "mi_per_well",
        "per_well_fit": RESULTS / "figures" / "top10_wells_scatter" / "mi_per_well_fit",
    },
}
for metric_dirs in DIRS.values():
    for d in metric_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

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

TOP_N = 10
CMAP  = cm.get_cmap("tab10")

MIN_FIT = 5   # minimum points to draw a per-well fit line


# ── Fig A: individual subplot per well ───────────────────────────────────────
def plot_per_well_subplots(gage_id, top_well_ids, metric_label, out_dir,
                          metric_vals=None):
    """metric_vals: dict {well_id -> metric_value} used for subtitle annotation."""
    gage_name = gage_names.get(gage_id, str(gage_id))
    gage_data = data[data["gage_id"] == gage_id]
    sub = gage_data[gage_data["well_id"].astype(str).isin(
        [str(w) for w in top_well_ids]
    )].dropna(subset=["delta_wte", "delta_q"])

    # Preserve ranking order (top_well_ids already sorted high→low by caller)
    present = set(sub["well_id"].unique())
    unique_wells = [w for w in top_well_ids if w in present]
    n_wells = len(unique_wells)
    if n_wells == 0:
        print(f"  Gage {gage_id}: no data – skipping.")
        return

    ncols = min(5, n_wells)
    nrows = math.ceil(n_wells / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 4.0 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    # Shared axis limits across all subplots for comparability
    all_x = sub["delta_wte"].values
    all_y = sub["delta_q"].values
    xpad = (all_x.max() - all_x.min()) * 0.05 or 1
    ypad = (all_y.max() - all_y.min()) * 0.05 or 1
    xlim = (all_x.min() - xpad, all_x.max() + xpad)
    ylim = (all_y.min() - ypad, all_y.max() + ypad)

    for i, wid in enumerate(unique_wells):
        ax = axes_flat[i]
        color = CMAP(i % 10)
        wdata = sub[sub["well_id"] == wid]
        x, y = wdata["delta_wte"].values, wdata["delta_q"].values

        ax.scatter(x, y, s=18, color=color, alpha=0.75, edgecolors="none", zorder=3)

        stats_lines = [f"n={len(x)}"]
        if len(x) >= MIN_FIT and x.std() > 0:
            slope, intercept, r_val, p_val, _ = linregress(x, y)
            x_fit = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(x_fit, slope * x_fit + intercept,
                    color="red", linewidth=1.4, zorder=4)
            p_str = f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.3f}"
            stats_lines += [f"slope={slope:.3f}", f"R²={r_val**2:.3f}", f"p={p_str}"]

        ax.text(0.97, 0.97, "\n".join(stats_lines),
                transform=ax.transAxes, fontsize=7.5,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.85, edgecolor="#AAAAAA"))

        ax.axhline(0, color="#CCCCCC", linewidth=0.6, linestyle=":", zorder=2)
        ax.axvline(0, color="#CCCCCC", linewidth=0.6, linestyle=":", zorder=2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("ΔWTE (ft)", fontsize=8)
        ax.set_ylabel("ΔQ (cfs)", fontsize=8)
        mv_str = f"  {metric_label}={metric_vals[str(wid)]:.3f}" if metric_vals and str(wid) in metric_vals else ""
        rank = unique_wells.index(wid) + 1
        ax.set_title(f"#{rank}  {str(wid)}{mv_str}",
                     fontsize=7.5, fontweight="bold")
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for j in range(n_wells, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Gage {gage_id}  —  {gage_name}\n"
        f"ΔQ vs ΔWTE per well  (top {n_wells} by {metric_label})",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out_path = out_dir / f"{gage_id}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [per_well]     Saved → {out_path}")


# ── Fig B: original combined scatter + per-well fit lines ────────────────────
def plot_combined_with_per_well_fits(gage_id, top_well_ids, metric_label, out_dir):
    gage_name = gage_names.get(gage_id, str(gage_id))
    gage_data = data[data["gage_id"] == gage_id]
    sub = gage_data[gage_data["well_id"].astype(str).isin(
        [str(w) for w in top_well_ids]
    )].dropna(subset=["delta_wte", "delta_q"])

    unique_wells = sub["well_id"].unique()
    n_wells = len(unique_wells)
    if n_wells == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    all_x = sub["delta_wte"].values
    all_y = sub["delta_q"].values
    xpad = (all_x.max() - all_x.min()) * 0.05 or 1
    xlim = (all_x.min() - xpad, all_x.max() + xpad)
    x_fit = np.linspace(xlim[0], xlim[1], 200)

    legend_entries = []
    for i, wid in enumerate(unique_wells):
        color = CMAP(i % 10)
        wdata = sub[sub["well_id"] == wid]
        x, y = wdata["delta_wte"].values, wdata["delta_q"].values

        sc = ax.scatter(x, y, s=22, color=color, alpha=0.65,
                        edgecolors="none", zorder=3)

        label = f"{str(wid)}"
        if len(x) >= MIN_FIT and x.std() > 0:
            slope, intercept, r_val, p_val, _ = linregress(x, y)
            ax.plot(x_fit, slope * x_fit + intercept,
                    color=color, linewidth=1.3, alpha=0.9, zorder=4)
            label += f"  (s={slope:.2f}, R²={r_val**2:.3f})"

        legend_entries.append((sc, label))

    # Overall regression (dashed black on top)
    slope_all, intercept_all, r_all, p_all, _ = linregress(all_x, all_y)
    ax.plot(x_fit, slope_all * x_fit + intercept_all,
            color="black", linewidth=2.0, linestyle="--",
            zorder=6, label="Overall fit")

    ax.axhline(0, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=2)
    ax.axvline(0, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=2)

    p_str = f"{p_all:.4f}" if p_all >= 0.0001 else "<0.0001"
    stats_text = (
        f"Overall (all wells pooled)\n"
        f"N={len(all_x)}  Wells={n_wells}\n"
        f"Slope={slope_all:.3f}\n"
        f"R²={r_all**2:.3f}\n"
        f"p={p_str}"
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8.5, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.88, edgecolor="#AAAAAA"))

    # Legend (per-well)
    handles = [e[0] for e in legend_entries]
    labels  = [e[1] for e in legend_entries]
    ax.legend(handles, labels, fontsize=7, loc="lower right",
              framealpha=0.85, title=f"Top {n_wells} wells by {metric_label}",
              title_fontsize=7.5)

    ax.set_xlabel("ΔWTE (ft)", fontsize=11)
    ax.set_ylabel("ΔQ (cfs)", fontsize=11)
    ax.set_title(
        f"Gage {gage_id}  —  {gage_name}\n"
        f"ΔQ vs ΔWTE  (top {n_wells} by {metric_label}, per-well fit lines)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = out_dir / f"{gage_id}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [per_well_fit] Saved → {out_path}")


# ── Run for R² ───────────────────────────────────────────────────────────────
print("=== Top 10 by R² ===")
for gage_id, grp in r2_df.groupby("gage_id"):
    print(f"\nGage {gage_id}:")
    ranked  = grp.nlargest(TOP_N, "r_squared")
    top_ids = ranked["well_id"].tolist()
    mvals   = dict(zip(ranked["well_id"].astype(str), ranked["r_squared"]))
    plot_per_well_subplots(gage_id, top_ids, "R²",
                           DIRS["r2"]["per_well"], metric_vals=mvals)
    plot_combined_with_per_well_fits(gage_id, top_ids, "R²",
                                     DIRS["r2"]["per_well_fit"])

# ── Run for MI ───────────────────────────────────────────────────────────────
print("\n=== Top 10 by MI ===")
for gage_id, grp in mi_df.groupby("gage_id"):
    print(f"\nGage {gage_id}:")
    ranked  = grp.nlargest(TOP_N, "mi")
    top_ids = ranked["well_id"].tolist()
    mvals   = dict(zip(ranked["well_id"].astype(str), ranked["mi"]))
    plot_per_well_subplots(gage_id, top_ids, "MI",
                           DIRS["mi"]["per_well"], metric_vals=mvals)
    plot_combined_with_per_well_fits(gage_id, top_ids, "MI",
                                     DIRS["mi"]["per_well_fit"])

# ── Summary tables ────────────────────────────────────────────────────────────
TAB_DIR = RESULTS / "figures" / "top10_wells_scatter" / "tables"
TAB_DIR.mkdir(exist_ok=True)

# Top-10 by R²: full regression stats, sorted rank within each gage
r2_tables = []
for gage_id, grp in r2_df.groupby("gage_id"):
    ranked = grp.nlargest(TOP_N, "r_squared").copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    r2_tables.append(ranked)

r2_top10 = pd.concat(r2_tables).reset_index(drop=True)
r2_top10 = r2_top10[["rank", "gage_id", "gage_name", "well_id",
                      "n_observations", "r_squared", "r_value", "p_value",
                      "slope", "intercept",
                      "pearson_r", "pearson_p", "spearman_r", "spearman_p"]].round(4)
r2_top10.to_csv(TAB_DIR / "top10_by_r2.csv", index=False)
print(f"  Saved → {TAB_DIR / 'top10_by_r2.csv'}")

# Top-10 by MI: MI stats + joined regression stats
mi_tables = []
for gage_id, grp in mi_df.groupby("gage_id"):
    ranked = grp.nlargest(TOP_N, "mi").copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    mi_tables.append(ranked)

mi_top10 = pd.concat(mi_tables).reset_index(drop=True)
mi_top10 = mi_top10[["rank", "gage_id", "gage_name", "well_id",
                      "n_records", "mi",
                      "pearson_r", "pearson_p", "spearman_r", "spearman_p"]]
r2_extra = r2_df[["well_id", "gage_id", "n_observations",
                   "r_squared", "slope", "p_value"]].rename(
    columns={"n_observations": "n_obs_reg", "r_squared": "r_squared_reg",
             "slope": "slope_reg", "p_value": "p_value_reg"})
mi_top10 = mi_top10.merge(r2_extra, on=["well_id", "gage_id"], how="left").round(4)
mi_top10.to_csv(TAB_DIR / "top10_by_mi.csv", index=False)
print(f"  Saved → {TAB_DIR / 'top10_by_mi.csv'}")

print(f"\nDone. New outputs:")
for metric, ds in DIRS.items():
    for kind, d in ds.items():
        print(f"  [{metric}/{kind}] → {d}")
