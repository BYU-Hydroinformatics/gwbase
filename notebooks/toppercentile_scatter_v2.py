"""
Extended top-percentile well scatter plots (v2).

Adds MI-based percentile plots and summary statistics tables for all groups.

New outputs (existing top10pct/ top25pct/ untouched):
  top10pct_per_well/        top10pct_per_well_fit/
  top25pct_per_well/        top25pct_per_well_fit/
  top10pct_mi_per_well/     top10pct_mi_per_well_fit/
  top25pct_mi_per_well/     top25pct_mi_per_well_fit/
  tables/  ← summary statistics CSV for every group
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
FIG_DIR = RESULTS / "figures" / "top_percentile_scatter"
TAB_DIR = FIG_DIR / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

PCTS = {"top10pct": 0.90, "top25pct": 0.75}
METRICS = {"r2": "r_squared", "mi": "mi"}   # friendly name → column

# Build all output dirs
OUT_DIRS = {}
for pct_key in PCTS:
    for metric_key in METRICS:
        suffix = "" if metric_key == "r2" else f"_{metric_key}"
        for kind in ("per_well", "per_well_fit"):
            key = (pct_key, metric_key, kind)
            d = FIG_DIR / f"{pct_key}{suffix}_{kind}"
            d.mkdir(parents=True, exist_ok=True)
            OUT_DIRS[key] = d

data  = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv")
data["date"] = pd.to_datetime(data["date"])
r2_df = pd.read_csv(RESULTS / "features" / "regression_by_well.csv")
mi_df = pd.read_csv(RESULTS / "features" / "mi_analysis.csv")
mi_df = mi_df[mi_df["mi"].notna()]

gage_names = (
    r2_df[["gage_id", "gage_name"]].dropna()
    .drop_duplicates("gage_id")
    .set_index("gage_id")["gage_name"]
    .to_dict()
)

CMAP    = cm.get_cmap("tab20")
MIN_FIT = 5

# ── helpers ────────────────────────────────────────────────────────────────────
def _well_stats(sub, wid):
    """Return per-well regression stats dict."""
    x = sub.loc[sub["well_id"] == wid, "delta_wte"].values
    y = sub.loc[sub["well_id"] == wid, "delta_q"].values
    rec = {"well_id": wid, "n_obs": len(x)}
    if len(x) >= MIN_FIT and x.std() > 0:
        slope, intercept, r_val, p_val, _ = linregress(x, y)
        rec.update(slope=slope, r2=r_val**2, p_val=p_val)
    else:
        rec.update(slope=np.nan, r2=np.nan, p_val=np.nan)
    return rec


# ── Fig A: individual subplot per well ────────────────────────────────────────
def plot_per_well_subplots(gage_id, top_well_ids, pct_label, n_total,
                           sort_metric_label, out_dir, metric_vals=None):
    """metric_vals: dict {well_id -> float} – used for ordering and title annotation."""
    gage_name = gage_names.get(gage_id, str(gage_id))
    sub = data[
        (data["gage_id"] == gage_id) &
        (data["well_id"].astype(str).isin([str(w) for w in top_well_ids]))
    ].dropna(subset=["delta_wte", "delta_q"])

    # Order wells high→low by metric; fall back to top_well_ids order
    present = set(sub["well_id"].unique())
    if metric_vals:
        unique_wells = sorted(
            [w for w in top_well_ids if w in present],
            key=lambda w: metric_vals.get(str(w), 0), reverse=True
        )
    else:
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

    all_x = sub["delta_wte"].values
    all_y = sub["delta_q"].values
    xpad = (all_x.max() - all_x.min()) * 0.05 or 1
    ypad = (all_y.max() - all_y.min()) * 0.05 or 1
    xlim = (all_x.min() - xpad, all_x.max() + xpad)
    ylim = (all_y.min() - ypad, all_y.max() + ypad)

    for i, wid in enumerate(unique_wells):
        ax = axes_flat[i]
        color = CMAP(i % 20)
        x = sub.loc[sub["well_id"] == wid, "delta_wte"].values
        y = sub.loc[sub["well_id"] == wid, "delta_q"].values

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
                transform=ax.transAxes, fontsize=7,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.85, edgecolor="#AAAAAA"))

        ax.axhline(0, color="#CCCCCC", linewidth=0.6, linestyle=":", zorder=2)
        ax.axvline(0, color="#CCCCCC", linewidth=0.6, linestyle=":", zorder=2)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("ΔWTE (ft)", fontsize=8)
        ax.set_ylabel("ΔQ (cfs)", fontsize=8)
        mv_str = f"  {sort_metric_label}={metric_vals[str(wid)]:.3f}" \
                 if metric_vals and str(wid) in metric_vals else ""
        rank = unique_wells.index(wid) + 1
        ax.set_title(f"#{rank}  {str(wid)}{mv_str}",
                     fontsize=7.5, fontweight="bold")
        ax.grid(True, alpha=0.2)

    for j in range(n_wells, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Gage {gage_id}  —  {gage_name}\n"
        f"ΔQ vs ΔWTE per well  "
        f"(top {pct_label} by {sort_metric_label},  {n_wells}/{n_total} wells)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out_path = out_dir / f"{gage_id}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    [per_well]     → {out_path.name}")


# ── Fig B: combined scatter + per-well fit lines ───────────────────────────────
def plot_combined_with_per_well_fits(gage_id, top_well_ids, pct_label, n_total,
                                     sort_metric_label, out_dir):
    gage_name = gage_names.get(gage_id, str(gage_id))
    sub = data[
        (data["gage_id"] == gage_id) &
        (data["well_id"].astype(str).isin([str(w) for w in top_well_ids]))
    ].dropna(subset=["delta_wte", "delta_q"])

    unique_wells = sub["well_id"].unique()
    n_wells = len(unique_wells)
    if n_wells == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    all_x = sub["delta_wte"].values
    all_y = sub["delta_q"].values
    xpad = (all_x.max() - all_x.min()) * 0.05 or 1
    xlim = (all_x.min() - xpad, all_x.max() + xpad)
    x_fit = np.linspace(xlim[0], xlim[1], 200)

    legend_entries = []
    for i, wid in enumerate(unique_wells):
        color = CMAP(i % 20)
        x = sub.loc[sub["well_id"] == wid, "delta_wte"].values
        y = sub.loc[sub["well_id"] == wid, "delta_q"].values

        sc = ax.scatter(x, y, s=18, color=color, alpha=0.60, edgecolors="none", zorder=3)

        label = f"{str(wid)}"
        if len(x) >= MIN_FIT and x.std() > 0:
            slope, intercept, r_val, p_val, _ = linregress(x, y)
            ax.plot(x_fit, slope * x_fit + intercept,
                    color=color, linewidth=1.2, alpha=0.85, zorder=4)
            label += f"  s={slope:.2f}  R²={r_val**2:.3f}"
        legend_entries.append((sc, label))

    slope_all, intercept_all, r_all, p_all, _ = linregress(all_x, all_y)
    ax.plot(x_fit, slope_all * x_fit + intercept_all,
            color="black", linewidth=2.0, linestyle="--", zorder=6)

    ax.axhline(0, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=2)
    ax.axvline(0, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=2)

    p_str = f"{p_all:.4f}" if p_all >= 0.0001 else "<0.0001"
    ax.text(0.98, 0.97,
            f"Overall (pooled)\nN={len(all_x)}  Wells={n_wells}/{n_total}\n"
            f"Slope={slope_all:.3f}\nR²={r_all**2:.3f}\np={p_str}",
            transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.88, edgecolor="#AAAAAA"))

    handles = [e[0] for e in legend_entries]
    labels  = [e[1] for e in legend_entries]
    ax.legend(handles, labels, fontsize=6.5, loc="lower right",
              framealpha=0.85, ncol=max(1, n_wells // 15 + 1),
              title=f"Top {pct_label} wells by {sort_metric_label}",
              title_fontsize=7)

    ax.set_xlabel("ΔWTE (ft)", fontsize=11)
    ax.set_ylabel("ΔQ (cfs)", fontsize=11)
    ax.set_title(
        f"Gage {gage_id}  —  {gage_name}\n"
        f"ΔQ vs ΔWTE  (top {pct_label} by {sort_metric_label}, per-well fit lines)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path = out_dir / f"{gage_id}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    [per_well_fit] → {out_path.name}")


# ── collect stats for table ────────────────────────────────────────────────────
def collect_group_stats(gage_id, top_well_ids, group_label):
    """Return list of per-well stat dicts and one pooled stat dict."""
    gage_name = gage_names.get(gage_id, str(gage_id))
    sub = data[
        (data["gage_id"] == gage_id) &
        (data["well_id"].astype(str).isin([str(w) for w in top_well_ids]))
    ].dropna(subset=["delta_wte", "delta_q"])

    rows = []
    for wid in sub["well_id"].unique():
        s = _well_stats(sub, wid)
        rows.append(dict(group=group_label, gage_id=gage_id,
                         gage_name=gage_name, level="well", **s))

    # pooled
    x_all = sub["delta_wte"].values
    y_all = sub["delta_q"].values
    if len(x_all) >= MIN_FIT and x_all.std() > 0:
        slope, _, r_val, p_val, _ = linregress(x_all, y_all)
        rows.append(dict(group=group_label, gage_id=gage_id,
                         gage_name=gage_name, level="pooled",
                         well_id="ALL", n_obs=len(x_all),
                         slope=slope, r2=r_val**2, p_val=p_val))
    return rows


# ── main loop ─────────────────────────────────────────────────────────────────
all_stats = []   # accumulate for final tables

for pct_key, quantile in PCTS.items():
    pct_label = pct_key.replace("top", "").replace("pct", "%")

    # ── R² ──────────────────────────────────────────────────────────────────
    print(f"\n=== {pct_label} by R² ===")
    for gage_id, grp in r2_df.groupby("gage_id"):
        n_total   = len(grp)
        threshold = grp["r_squared"].quantile(quantile)
        ranked    = grp[grp["r_squared"] >= threshold].sort_values("r_squared", ascending=False)
        top_ids   = ranked["well_id"].tolist()
        mvals     = dict(zip(ranked["well_id"].astype(str), ranked["r_squared"]))
        group_lbl = f"{pct_label}_R2"
        print(f"  Gage {gage_id}: {len(top_ids)}/{n_total} wells (R²≥{threshold:.4f})")

        plot_per_well_subplots(gage_id, top_ids, pct_label, n_total, "R²",
                               OUT_DIRS[(pct_key, "r2", "per_well")], metric_vals=mvals)
        plot_combined_with_per_well_fits(gage_id, top_ids, pct_label, n_total, "R²",
                                         OUT_DIRS[(pct_key, "r2", "per_well_fit")])
        all_stats.extend(collect_group_stats(gage_id, top_ids, group_lbl))

    # ── MI ──────────────────────────────────────────────────────────────────
    print(f"\n=== {pct_label} by MI ===")
    for gage_id, grp in mi_df.groupby("gage_id"):
        n_total   = len(grp)
        threshold = grp["mi"].quantile(quantile)
        ranked    = grp[grp["mi"] >= threshold].sort_values("mi", ascending=False)
        top_ids   = ranked["well_id"].tolist()
        mvals     = dict(zip(ranked["well_id"].astype(str), ranked["mi"]))
        group_lbl = f"{pct_label}_MI"
        print(f"  Gage {gage_id}: {len(top_ids)}/{n_total} wells (MI≥{threshold:.4f})")

        plot_per_well_subplots(gage_id, top_ids, pct_label, n_total, "MI",
                               OUT_DIRS[(pct_key, "mi", "per_well")], metric_vals=mvals)
        plot_combined_with_per_well_fits(gage_id, top_ids, pct_label, n_total, "MI",
                                         OUT_DIRS[(pct_key, "mi", "per_well_fit")])
        all_stats.extend(collect_group_stats(gage_id, top_ids, group_lbl))

# ── also collect stats for top-N groups from top10_scatter_v2 ─────────────────
TOP_N = 10
print(f"\n=== Top {TOP_N} by R² (absolute N) ===")
for gage_id, grp in r2_df.groupby("gage_id"):
    top_ids = grp.nlargest(TOP_N, "r_squared")["well_id"].tolist()
    all_stats.extend(collect_group_stats(gage_id, top_ids, f"top{TOP_N}N_R2"))

print(f"=== Top {TOP_N} by MI (absolute N) ===")
for gage_id, grp in mi_df.groupby("gage_id"):
    top_ids = grp.nlargest(TOP_N, "mi")["well_id"].tolist()
    all_stats.extend(collect_group_stats(gage_id, top_ids, f"top{TOP_N}N_MI"))

# ── write tables ──────────────────────────────────────────────────────────────
stats_df = pd.DataFrame(all_stats)

# 1. Full per-well table
stats_df.round(4).to_csv(TAB_DIR / "all_groups_per_well_stats.csv", index=False)
print(f"\nSaved: tables/all_groups_per_well_stats.csv  ({len(stats_df)} rows)")

# 2. Pooled summary (one row per group × gage)
pooled = stats_df[stats_df["level"] == "pooled"].copy()
pooled.drop(columns=["level", "well_id"]).round(4).to_csv(
    TAB_DIR / "all_groups_pooled_stats.csv", index=False)
print(f"Saved: tables/all_groups_pooled_stats.csv  ({len(pooled)} rows)")

# 3. Per-well summary pivoted: median slope / R² / n_obs per group × gage
well_only = stats_df[stats_df["level"] == "well"].copy()
summary = (
    well_only.groupby(["group", "gage_id", "gage_name"])
    .agg(
        n_wells      = ("well_id",  "count"),
        median_n_obs = ("n_obs",    "median"),
        mean_slope   = ("slope",    "mean"),
        median_slope = ("slope",    "median"),
        mean_r2      = ("r2",       "mean"),
        median_r2    = ("r2",       "median"),
        pct_pos_slope= ("slope",    lambda s: (s > 0).mean() * 100),
    )
    .round(4)
    .reset_index()
)
summary.to_csv(TAB_DIR / "all_groups_well_summary.csv", index=False)
print(f"Saved: tables/all_groups_well_summary.csv  ({len(summary)} rows)")

# Print readable pivot
print("\n── Median R² by group × gage ──")
pivot_r2 = summary.pivot_table(index="gage_id", columns="group", values="median_r2").round(4)
print(pivot_r2.to_string())
print("\n── Median slope by group × gage ──")
pivot_sl = summary.pivot_table(index="gage_id", columns="group", values="median_slope").round(3)
print(pivot_sl.to_string())

print(f"\nAll tables → {TAB_DIR}")
print(f"All figures → {FIG_DIR}")
