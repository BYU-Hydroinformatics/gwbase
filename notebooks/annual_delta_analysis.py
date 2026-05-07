"""
Annual delta analysis: ΔQ_annual vs ΔWTE_annual regression.

For each well-gage pair:
  1. Compute annual mean WTE and Q (from monthly PCHIP data, bfd=1 already)
  2. First-difference year-over-year: ΔWTE = WTE_t - WTE_{t-1}
  3. Regress ΔQ ~ ΔWTE at the annual scale

Outputs (all under results/annual_delta/):
  features/
    data_annual_deltas.csv       — per well-gage-year annual means + deltas
    regression_annual_by_well.csv — per-well regression stats
  figures/
    slope_distribution.png       — annual vs monthly slope sign comparison
    scatter_by_gage/             — ΔQ vs ΔWTE scatter, all wells pooled per gage
    top10_by_r2/                 — top-10 R² wells per gage (subplots)
    top10_by_r2_fit/             — top-10 combined scatter + per-well fit lines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from scipy.stats import linregress, spearmanr, pearsonr
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
ANN_DIR = RESULTS / "annual_delta"

# Output directories
FEAT_DIR  = ANN_DIR / "features"
FIG_DIR   = ANN_DIR / "figures"
SCAT_DIR  = FIG_DIR / "scatter_by_gage"
T10_DIR   = FIG_DIR / "top10_by_r2"
T10F_DIR  = FIG_DIR / "top10_by_r2_fit"
for d in [FEAT_DIR, FIG_DIR, SCAT_DIR, T10_DIR, T10F_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MIN_OBS  = 8    # minimum years of paired data
MIN_FIT  = 5    # minimum points to draw a fit line
TOP_N    = 10
CMAP     = cm.get_cmap("tab10")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
data = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv",
                   parse_dates=["date"])
data["well_id"] = data["well_id"].astype(str)
data["gage_id"] = data["gage_id"].astype(str)

# ── Step 1: Annual means ──────────────────────────────────────────────────────
print("Computing annual means...")
data["year"] = data["date"].dt.year

annual = (data.groupby(["well_id", "gage_id", "gage_name", "year"])
          .agg(
              wte_ann   = ("wte",  "mean"),
              q_ann     = ("q",    "mean"),
              n_months  = ("wte",  "count"),
              well_lat  = ("well_lat",  "first"),
              well_lon  = ("well_lon",  "first"),
              delta_elev= ("delta_elev","first"),
              delta_bin = ("delta_bin", "first"),
          )
          .reset_index()
          .sort_values(["well_id", "gage_id", "year"]))

# Require at least 6 months of data in a year to trust the annual mean
annual = annual[annual["n_months"] >= 6].copy()

# ── Step 2: Year-over-year first differences ──────────────────────────────────
annual["delta_wte_ann"] = annual.groupby(["well_id","gage_id"])["wte_ann"].diff()
annual["delta_q_ann"]   = annual.groupby(["well_id","gage_id"])["q_ann"].diff()

# Keep only consecutive years (diff = 1)
annual["year_diff"] = annual.groupby(["well_id","gage_id"])["year"].diff()
annual = annual[annual["year_diff"] == 1].dropna(subset=["delta_wte_ann","delta_q_ann"])

# ── Remove delta_wte outliers per well-gage (IQR × 3) ────────────────────────
def iqr_mask(series, k=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series >= q1 - k*iqr) & (series <= q3 + k*iqr)

before = len(annual)
annual = annual[annual.groupby(["well_id","gage_id"])["delta_wte_ann"]
                .transform(iqr_mask)].copy()
after = len(annual)
print(f"  Outlier removal (IQR×3 on ΔWTE): {before - after} rows removed")

annual.to_csv(FEAT_DIR / "data_annual_deltas.csv", index=False)
print(f"  Saved data_annual_deltas.csv  ({len(annual)} rows, "
      f"{annual['well_id'].nunique()} wells)")

# ── Step 3: Per-well regression ───────────────────────────────────────────────
print("Running per-well regressions...")
reg_rows = []
for (wid, gid, gname), grp in annual.groupby(["well_id","gage_id","gage_name"]):
    x = grp["delta_wte_ann"].values
    y = grp["delta_q_ann"].values
    if len(x) < MIN_OBS or x.std() == 0:
        continue
    s, intercept, r_val, p_val, std_err = linregress(x, y)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    reg_rows.append({
        "well_id": wid, "gage_id": gid, "gage_name": gname,
        "n_years": len(x),
        "slope": round(s, 4), "intercept": round(intercept, 4),
        "r_squared": round(r_val**2, 4), "r_value": round(r_val, 4),
        "p_value": round(p_val, 6),
        "pearson_r": round(pr, 4), "pearson_p": round(pp, 6),
        "spearman_r": round(sr, 4), "spearman_p": round(sp, 6),
    })

reg_df = pd.DataFrame(reg_rows)
reg_df.to_csv(FEAT_DIR / "regression_annual_by_well.csv", index=False)
print(f"  Saved regression_annual_by_well.csv  ({len(reg_df)} wells)")

# Summary per gage
print("\n=== Annual regression summary per gage ===")
print(f"{'Gage':<46} {'Wells':>6} {'%Neg':>6} {'%Sig':>6} "
      f"{'Median slope':>13} {'Median R²':>10}")
print("-"*90)
for gname, grp in reg_df.groupby("gage_name"):
    pneg = 100 * (grp["slope"] < 0).mean()
    psig = 100 * (grp["p_value"] < 0.05).mean()
    short = gname.split("NEAR")[0].split("AT")[0].strip()[:44]
    print(f"{short:<46} {len(grp):>6} {pneg:>5.0f}% {psig:>5.0f}% "
          f"{grp['slope'].median():>13.3f} {grp['r_squared'].median():>10.3f}")

# ── Step 4: Fig A — slope distribution comparison ────────────────────────────
print("\nGenerating figures...")
monthly_reg = pd.read_csv(RESULTS / "features" / "regression_by_well.csv")
monthly_reg["well_id"] = monthly_reg["well_id"].astype(str)

gage_order = [
    "BEAR RIVER NEAR CORINNE - UT",
    "WEBER RIVER NEAR PLAIN CITY - UT",
    "PROVO RIVER AT PROVO - UT",
    "SPANISH FORK NEAR LAKE SHORE - UTAH",
    "LITTLE COTTONWOOD CREEK @ JORDAN RIVER NR SLC",
]
gage_short = {
    "BEAR RIVER NEAR CORINNE - UT":                  "Bear River",
    "WEBER RIVER NEAR PLAIN CITY - UT":               "Weber River",
    "PROVO RIVER AT PROVO - UT":                      "Provo River",
    "SPANISH FORK NEAR LAKE SHORE - UTAH":            "Spanish Fork",
    "LITTLE COTTONWOOD CREEK @ JORDAN RIVER NR SLC":  "Little Cottonwood",
}

fig, axes = plt.subplots(5, 2, figsize=(13, 16),
                          gridspec_kw={"hspace": 0.55, "wspace": 0.35})
fig.suptitle("Slope distribution: Monthly delta vs Annual delta\n"
             "Red = negative slope,  Blue = positive slope",
             fontsize=13, fontweight="bold")

for row, gname in enumerate(gage_order):
    m_sub = monthly_reg[monthly_reg["gage_name"] == gname]["slope"].values
    a_sub = reg_df[reg_df["gage_name"] == gname]["slope"].values
    gshort = gage_short[gname]

    for col, (slopes, label) in enumerate([(m_sub, "Monthly delta"),
                                            (a_sub, "Annual delta")]):
        ax = axes[row, col]
        if len(slopes) == 0:
            ax.set_visible(False)
            continue
        n_neg = (slopes < 0).sum()
        n_pos = (slopes > 0).sum()
        p2, p98 = np.percentile(slopes, 2), np.percentile(slopes, 98)
        slopes_c = np.clip(slopes, p2, p98)
        bins = min(30, max(8, len(slopes) // 3))
        counts, edges, patches = ax.hist(slopes_c, bins=bins,
                                          edgecolor="white", linewidth=0.3)
        for patch, left in zip(patches, edges[:-1]):
            patch.set_facecolor("#E24A33" if left < 0 else "#4C72B0")
            patch.set_alpha(0.75)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(
            f"{gshort} — {label}\n"
            f"neg {n_neg} ({100*n_neg/len(slopes):.0f}%)  "
            f"pos {n_pos} ({100*n_pos/len(slopes):.0f}%)",
            fontsize=8, fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("Count", fontsize=8)
        if row == 4:
            ax.set_xlabel("Slope (cfs/ft·yr)", fontsize=8)

p = FIG_DIR / "slope_distribution.png"
plt.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved → {p}")

# ── Step 5: Fig B — pooled scatter per gage ───────────────────────────────────
for gname in gage_order:
    gshort = gage_short[gname]
    sub = annual[annual["gage_name"] == gname].dropna(
        subset=["delta_wte_ann","delta_q_ann"])
    if len(sub) < 5:
        continue

    x, y = sub["delta_wte_ann"].values, sub["delta_q_ann"].values
    fig, ax = plt.subplots(figsize=(8, 5))

    # Colour by well
    wells = sub["well_id"].unique()
    for i, wid in enumerate(wells):
        wd = sub[sub["well_id"] == wid]
        ax.scatter(wd["delta_wte_ann"], wd["delta_q_ann"],
                   s=25, color=CMAP(i % 10), alpha=0.6,
                   edgecolors="none", zorder=3)

    # Overall fit
    s, intercept, r_val, p_val, _ = linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_fit, s * x_fit + intercept,
            color="black", linewidth=2, linestyle="--", zorder=5)

    ax.axhline(0, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=2)
    ax.axvline(0, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=2)

    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    ax.text(0.98, 0.97,
            f"N={len(x)}  Wells={len(wells)}\n"
            f"Slope={s:.3f}\nR²={r_val**2:.3f}\np={p_str}",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.88, edgecolor="#AAAAAA"))

    ax.set_xlabel("ΔWTE annual (ft/yr)", fontsize=11)
    ax.set_ylabel("ΔQ annual (cfs/yr)", fontsize=11)
    ax.set_title(f"{gname}\nΔQ vs ΔWTE — Annual delta (all wells)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = SCAT_DIR / f"{gname.split()[0].lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

# ── Step 6: Fig C — top-10 R² per-well subplots ──────────────────────────────
def plot_per_well_subplots(gage_name, top_ids, out_dir, metric_vals=None):
    gshort = gage_short.get(gage_name, gage_name)
    sub = annual[(annual["gage_name"] == gage_name) &
                 (annual["well_id"].isin([str(w) for w in top_ids]))
                 ].dropna(subset=["delta_wte_ann","delta_q_ann"])
    present = set(sub["well_id"].unique())
    ordered = [w for w in top_ids if w in present]
    if not ordered:
        return
    n = len(ordered)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.2*ncols, 4.0*nrows), squeeze=False)
    axes_flat = axes.flatten()

    all_x = sub["delta_wte_ann"].values
    all_y = sub["delta_q_ann"].values
    xpad = (all_x.max()-all_x.min())*0.05 or 1
    ypad = (all_y.max()-all_y.min())*0.05 or 1
    xlim = (all_x.min()-xpad, all_x.max()+xpad)
    ylim = (all_y.min()-ypad, all_y.max()+ypad)

    for i, wid in enumerate(ordered):
        ax = axes_flat[i]
        color = CMAP(i % 10)
        wd = sub[sub["well_id"] == wid]
        x, y = wd["delta_wte_ann"].values, wd["delta_q_ann"].values
        ax.scatter(x, y, s=30, color=color, alpha=0.8,
                   edgecolors="none", zorder=3)
        stats = [f"n={len(x)}yr"]
        if len(x) >= MIN_FIT and x.std() > 0:
            s, ic, r_val, p_val, _ = linregress(x, y)
            x_fit = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(x_fit, s*x_fit+ic, color="red",
                    linewidth=1.4, zorder=4)
            p_str = f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.3f}"
            stats += [f"slope={s:.3f}", f"R²={r_val**2:.3f}", f"p={p_str}"]
        ax.text(0.97, 0.97, "\n".join(stats),
                transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.85, edgecolor="#AAAAAA"))
        ax.axhline(0, color="#CCCCCC", linewidth=0.6, linestyle=":")
        ax.axvline(0, color="#CCCCCC", linewidth=0.6, linestyle=":")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("ΔWTE (ft/yr)", fontsize=8)
        ax.set_ylabel("ΔQ (cfs/yr)", fontsize=8)
        mv_str = (f"  R²={metric_vals[str(wid)]:.3f}"
                  if metric_vals and str(wid) in metric_vals else "")
        ax.set_title(f"#{i+1}  {str(wid)}{mv_str}", fontsize=7.5, fontweight="bold")
        ax.grid(True, alpha=0.2)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"{gage_name}\nΔQ vs ΔWTE per well — Annual delta (top {n} by R²)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = out_dir / f"{gage_name.split()[0].lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_combined_fit(gage_name, top_ids, out_dir):
    gshort = gage_short.get(gage_name, gage_name)
    sub = annual[(annual["gage_name"] == gage_name) &
                 (annual["well_id"].isin([str(w) for w in top_ids]))
                 ].dropna(subset=["delta_wte_ann","delta_q_ann"])
    wells = sub["well_id"].unique()
    if len(wells) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    all_x = sub["delta_wte_ann"].values
    all_y = sub["delta_q_ann"].values
    xpad = (all_x.max()-all_x.min())*0.05 or 1
    xlim = (all_x.min()-xpad, all_x.max()+xpad)
    x_fit = np.linspace(xlim[0], xlim[1], 200)
    legend_entries = []
    for i, wid in enumerate(wells):
        color = CMAP(i % 10)
        wd = sub[sub["well_id"] == wid]
        x, y = wd["delta_wte_ann"].values, wd["delta_q_ann"].values
        sc = ax.scatter(x, y, s=28, color=color, alpha=0.7,
                        edgecolors="none", zorder=3)
        label = str(wid)
        if len(x) >= MIN_FIT and x.std() > 0:
            s, ic, r_val, p_val, _ = linregress(x, y)
            ax.plot(x_fit, s*x_fit+ic, color=color,
                    linewidth=1.3, alpha=0.9, zorder=4)
            label += f"  (s={s:.2f}, R²={r_val**2:.3f})"
        legend_entries.append((sc, label))
    # Overall fit
    s_all, ic_all, r_all, p_all, _ = linregress(all_x, all_y)
    ax.plot(x_fit, s_all*x_fit+ic_all, color="black",
            linewidth=2, linestyle="--", zorder=6, label="Overall fit")
    ax.axhline(0, color="#CCCCCC", linewidth=0.7, linestyle=":")
    ax.axvline(0, color="#CCCCCC", linewidth=0.7, linestyle=":")
    p_str = f"{p_all:.4f}" if p_all >= 0.0001 else "<0.0001"
    ax.text(0.98, 0.97,
            f"Overall (pooled)\nN={len(all_x)}  Wells={len(wells)}\n"
            f"Slope={s_all:.3f}\nR²={r_all**2:.3f}\np={p_str}",
            transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.88, edgecolor="#AAAAAA"))
    handles = [e[0] for e in legend_entries]
    labels  = [e[1] for e in legend_entries]
    ax.legend(handles, labels, fontsize=7, loc="lower right",
              framealpha=0.85, title=f"Top {len(wells)} wells by R²",
              title_fontsize=7.5)
    ax.set_xlabel("ΔWTE annual (ft/yr)", fontsize=11)
    ax.set_ylabel("ΔQ annual (cfs/yr)", fontsize=11)
    ax.set_title(f"{gage_name}\nΔQ vs ΔWTE — Annual delta (top {len(wells)} by R², per-well fit)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = out_dir / f"{gage_name.split()[0].lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")


for gname, grp in reg_df.groupby("gage_name"):
    ranked = grp.nlargest(TOP_N, "r_squared")
    top_ids = ranked["well_id"].tolist()
    mvals   = dict(zip(ranked["well_id"].astype(str), ranked["r_squared"]))
    plot_per_well_subplots(gname, top_ids, T10_DIR, metric_vals=mvals)
    plot_combined_fit(gname, top_ids, T10F_DIR)

# ── Summary table ─────────────────────────────────────────────────────────────
tab_dir = ANN_DIR / "figures" / "tables"
tab_dir.mkdir(exist_ok=True)
tables = []
for gname, grp in reg_df.groupby("gage_name"):
    ranked = grp.nlargest(TOP_N, "r_squared").copy()
    ranked.insert(0, "rank", range(1, len(ranked)+1))
    tables.append(ranked)
top10 = pd.concat(tables).reset_index(drop=True)
top10.to_csv(tab_dir / "top10_annual_by_r2.csv", index=False)
print(f"\n  Saved → {tab_dir / 'top10_annual_by_r2.csv'}")

print(f"\nAll outputs → {ANN_DIR}")
