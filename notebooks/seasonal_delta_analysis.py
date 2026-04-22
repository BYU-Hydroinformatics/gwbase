"""
Seasonal delta analysis: ΔQ vs ΔWTE scatter plots by season.

Method:
  - Split monthly data into 4 seasons
  - For each well-gage-season, compute year-over-year delta within that season
    (e.g., Winter 2005 mean minus Winter 2004 mean)
  - Minimum 3 years / 20 observations per well-gage pair (matching config)
  - Plot per-gage 4-panel scatter (one panel per season)

Seasons:
  Winter : Dec, Jan, Feb  (DJF)
  Spring : Mar, Apr, May  (MAM)
  Summer : Jun, Jul, Aug  (JJA)
  Fall   : Sep, Oct, Nov  (SON)

Outputs (all under results/seasonal_delta/):
  features/
    data_seasonal_deltas.csv
    regression_seasonal_by_well.csv
  figures/
    scatter_by_gage/      — one PNG per gage (2×2 season subplots)
    top10_by_r2/          — top-10 R² per gage per season (subplots)
    tables/
      top10_seasonal_by_r2.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from scipy.stats import linregress, pearsonr, spearmanr
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "seasonal_delta"

FEAT_DIR = OUT_DIR / "features"
FIG_DIR  = OUT_DIR / "figures"
SCAT_DIR = FIG_DIR / "scatter_by_gage"
T10_DIR  = FIG_DIR / "top10_by_r2"
TAB_DIR  = FIG_DIR / "tables"
for d in [FEAT_DIR, FIG_DIR, SCAT_DIR, T10_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MIN_OBS   = 20   # minimum season-year pairs per well-gage
MIN_YEARS = 3    # minimum years of data
MIN_FIT   = 5
TOP_N     = 10
CMAP      = cm.get_cmap("tab10")

SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring",  4: "Spring", 5: "Spring",
              6: "Summer",  7: "Summer", 8: "Summer",
              9: "Fall",   10: "Fall",  11: "Fall"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
SEASON_COLOR = {"Winter": "#4C72B0", "Spring": "#55A868",
                "Summer": "#E24A33", "Fall":   "#C4AD3A"}

GAGE_SHORT = {
    "BEAR RIVER NEAR CORINNE - UT":                  "Bear River",
    "WEBER RIVER NEAR PLAIN CITY - UT":               "Weber River",
    "PROVO RIVER AT PROVO - UT":                      "Provo River",
    "SPANISH FORK NEAR LAKE SHORE - UTAH":            "Spanish Fork",
    "LITTLE COTTONWOOD CREEK @ JORDAN RIVER NR SLC":  "Little Cottonwood",
}

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
data = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv",
                   parse_dates=["date"])
data["well_id"] = data["well_id"].astype(str)
data["gage_id"] = data["gage_id"].astype(str)

data["month"]  = data["date"].dt.month
data["season"] = data["month"].map(SEASON_MAP)
# Assign season-year: Dec belongs to next year's Winter
data["season_year"] = data["date"].dt.year
data.loc[data["month"] == 12, "season_year"] += 1

# ── Step 1: Seasonal means per well-gage-season-year ─────────────────────────
print("Computing seasonal means...")
seas_mean = (data.groupby(["well_id","gage_id","gage_name","season","season_year"])
             .agg(wte_seas=("wte","mean"),
                  q_seas  =("q",  "mean"),
                  n_months=("wte","count"))
             .reset_index()
             .sort_values(["well_id","gage_id","season","season_year"]))

# Require all 3 months present in season
seas_mean = seas_mean[seas_mean["n_months"] >= 3].copy()

# ── Step 2: Year-over-year delta within each season ──────────────────────────
seas_mean["delta_wte_seas"] = seas_mean.groupby(
    ["well_id","gage_id","season"])["wte_seas"].diff()
seas_mean["delta_q_seas"] = seas_mean.groupby(
    ["well_id","gage_id","season"])["q_seas"].diff()
seas_mean["year_diff"] = seas_mean.groupby(
    ["well_id","gage_id","season"])["season_year"].diff()

# Keep consecutive years only
seas_mean = seas_mean[seas_mean["year_diff"] == 1].dropna(
    subset=["delta_wte_seas","delta_q_seas"])

# ── Outlier removal per well-gage-season (IQR×3) ─────────────────────────────
def iqr_mask(series, k=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(True, index=series.index)
    return (series >= q1 - k*iqr) & (series <= q3 + k*iqr)

before = len(seas_mean)
seas_mean = seas_mean[seas_mean.groupby(["well_id","gage_id","season"])[
    "delta_wte_seas"].transform(iqr_mask)].copy()
print(f"  Outlier removal (IQR×3): {before - len(seas_mean)} rows removed")

seas_mean.to_csv(FEAT_DIR / "data_seasonal_deltas.csv", index=False)
print(f"  Saved data_seasonal_deltas.csv  ({len(seas_mean)} rows)")

# ── Step 3: Per-well-season regression ───────────────────────────────────────
print("Running per-well regressions...")
reg_rows = []
for (wid, gid, gname, season), grp in seas_mean.groupby(
        ["well_id","gage_id","gage_name","season"]):
    x = grp["delta_wte_seas"].values
    y = grp["delta_q_seas"].values
    n_years = grp["season_year"].nunique()
    if len(x) < MIN_OBS or n_years < MIN_YEARS or x.std() == 0:
        continue
    s, intercept, r_val, p_val, _ = linregress(x, y)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    reg_rows.append({
        "well_id": wid, "gage_id": gid, "gage_name": gname, "season": season,
        "n_obs": len(x), "n_years": n_years,
        "slope": round(s,4), "intercept": round(intercept,4),
        "r_squared": round(r_val**2,4), "r_value": round(r_val,4),
        "p_value": round(p_val,6),
        "pearson_r": round(pr,4), "pearson_p": round(pp,6),
        "spearman_r": round(sr,4), "spearman_p": round(sp,6),
    })

reg_df = pd.DataFrame(reg_rows)
reg_df.to_csv(FEAT_DIR / "regression_seasonal_by_well.csv", index=False)
print(f"  Saved regression_seasonal_by_well.csv  ({len(reg_df)} rows)")

# Summary
print("\n=== Seasonal regression summary ===")
print(f"{'Gage':<20} {'Season':<8} {'Wells':>6} {'%Neg':>6} "
      f"{'%Sig':>6} {'Median slope':>13} {'Median R²':>10}")
print("-"*75)
for (gname, season), grp in reg_df.groupby(["gage_name","season"]):
    short = GAGE_SHORT.get(gname, gname)[:18]
    pneg = 100*(grp["slope"]<0).mean()
    psig = 100*(grp["p_value"]<0.05).mean()
    print(f"{short:<20} {season:<8} {len(grp):>6} {pneg:>5.0f}% "
          f"{psig:>5.0f}% {grp['slope'].median():>13.3f} "
          f"{grp['r_squared'].median():>10.3f}")

# ── Step 4: Fig A — per-gage 2×2 scatter (all wells pooled) ──────────────────
print("\nGenerating scatter figures...")

def pooled_stats(df, xcol, ycol):
    sub = df[[xcol,ycol]].dropna()
    x, y = sub[xcol].values, sub[ycol].values
    if len(x) < 5 or x.std() == 0:
        return None
    s, ic, r_val, p_val, _ = linregress(x, y)
    return {"slope":s, "r2":r_val**2, "p":p_val,
            "n":len(x), "n_wells":df["well_id"].nunique(),
            "intercept":ic}

for gname in GAGE_SHORT:
    gshort = GAGE_SHORT[gname]
    gdata  = seas_mean[seas_mean["gage_name"] == gname]
    if gdata.empty:
        continue

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                              gridspec_kw={"hspace":0.45, "wspace":0.35})
    fig.suptitle(f"{gname}\nΔQ vs ΔWTE — Seasonal delta (all wells)",
                 fontsize=12, fontweight="bold")
    axes_flat = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

    for ax, season in zip(axes_flat, SEASON_ORDER):
        sub = gdata[gdata["season"]==season].dropna(
            subset=["delta_wte_seas","delta_q_seas"])
        color = SEASON_COLOR[season]

        if len(sub) < 5:
            ax.set_title(f"{season}\n(insufficient data)", fontsize=10)
            ax.set_visible(True)
            continue

        # scatter coloured by well
        wells = sub["well_id"].unique()
        for i, wid in enumerate(wells):
            wd = sub[sub["well_id"]==wid]
            ax.scatter(wd["delta_wte_seas"], wd["delta_q_seas"],
                       s=20, color=CMAP(i % 10), alpha=0.55,
                       edgecolors="none", zorder=3)

        # pooled fit
        st = pooled_stats(sub, "delta_wte_seas", "delta_q_seas")
        if st:
            xv = sub["delta_wte_seas"].values
            x_fit = np.linspace(xv.min(), xv.max(), 200)
            ax.plot(x_fit, st["slope"]*x_fit + st["intercept"],
                    color="black", linewidth=1.8, linestyle="--", zorder=5)
            p_str = f"{st['p']:.4f}" if st['p'] >= 0.0001 else "<0.0001"
            ax.text(0.98, 0.97,
                    f"N={st['n']}  Wells={st['n_wells']}\n"
                    f"Slope={st['slope']:.3f}\n"
                    f"R²={st['r2']:.3f}\np={p_str}",
                    transform=ax.transAxes, fontsize=8.5,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white",
                              alpha=0.88, edgecolor="#AAAAAA"))

        ax.axhline(0, color="#CCCCCC", linewidth=0.7, linestyle=":")
        ax.axvline(0, color="#CCCCCC", linewidth=0.7, linestyle=":")
        ax.set_xlabel("ΔWTE seasonal (ft/yr)", fontsize=9)
        ax.set_ylabel("ΔQ seasonal (cfs/yr)", fontsize=9)
        ax.set_title(season, fontsize=11, fontweight="bold",
                     color=SEASON_COLOR[season])
        ax.grid(True, alpha=0.2)

    out = SCAT_DIR / f"{gname.split()[0].lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

# ── Step 5: Fig B — top-10 R² per-well subplots, one fig per gage per season ──
print("\nGenerating top-10 figures...")

for (gname, season), grp in reg_df.groupby(["gage_name","season"]):
    gshort = GAGE_SHORT.get(gname, gname)
    ranked = grp.nlargest(TOP_N, "r_squared")
    top_ids = ranked["well_id"].tolist()
    mvals   = dict(zip(ranked["well_id"].astype(str), ranked["r_squared"]))

    sub = seas_mean[(seas_mean["gage_name"]==gname) &
                    (seas_mean["season"]==season) &
                    (seas_mean["well_id"].isin(top_ids))
                    ].dropna(subset=["delta_wte_seas","delta_q_seas"])
    present = [w for w in top_ids if w in set(sub["well_id"].unique())]
    if not present:
        continue

    n = len(present)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.2*ncols, 4.0*nrows), squeeze=False)
    axes_flat = axes.flatten()

    all_x = sub["delta_wte_seas"].values
    all_y = sub["delta_q_seas"].values
    xpad  = (all_x.max()-all_x.min())*0.05 or 1
    ypad  = (all_y.max()-all_y.min())*0.05 or 1
    xlim  = (all_x.min()-xpad, all_x.max()+xpad)
    ylim  = (all_y.min()-ypad, all_y.max()+ypad)

    for i, wid in enumerate(present):
        ax = axes_flat[i]
        color = CMAP(i % 10)
        wd = sub[sub["well_id"]==wid]
        x, y = wd["delta_wte_seas"].values, wd["delta_q_seas"].values
        ax.scatter(x, y, s=28, color=color, alpha=0.8,
                   edgecolors="none", zorder=3)
        stats = [f"n={len(x)}"]
        if len(x) >= MIN_FIT and x.std() > 0:
            s, ic, r_val, p_val, _ = linregress(x, y)
            x_fit = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(x_fit, s*x_fit+ic, color="red", linewidth=1.4, zorder=4)
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
        mv = mvals.get(str(wid), None)
        mv_str = f"  R²={mv:.3f}" if mv is not None else ""
        ax.set_title(f"#{i+1}  {str(wid)}{mv_str}", fontsize=7.5, fontweight="bold")
        ax.grid(True, alpha=0.2)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{gname} — {season}\n"
                 f"ΔQ vs ΔWTE per well (top {n} by R², seasonal delta)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = T10_DIR / f"{gname.split()[0].lower()}_{season.lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
tables = []
for (gname, season), grp in reg_df.groupby(["gage_name","season"]):
    ranked = grp.nlargest(TOP_N, "r_squared").copy()
    ranked.insert(0, "rank", range(1, len(ranked)+1))
    tables.append(ranked)
top10 = pd.concat(tables).reset_index(drop=True)
top10.to_csv(TAB_DIR / "top10_seasonal_by_r2.csv", index=False)
print(f"\n  Saved → {TAB_DIR / 'top10_seasonal_by_r2.csv'}")
print(f"\nAll outputs → {OUT_DIR}")
