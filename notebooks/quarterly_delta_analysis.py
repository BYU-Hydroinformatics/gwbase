"""
Quarterly delta analysis: ΔQ vs ΔWTE scatter plots by calendar quarter.

Method:
  - Split monthly data into 4 calendar quarters
      Q1 : Jan, Feb, Mar
      Q2 : Apr, May, Jun
      Q3 : Jul, Aug, Sep
      Q4 : Oct, Nov, Dec
  - For each well-gage-quarter, compute year-over-year delta within that quarter
    (e.g., Q1 2005 mean minus Q1 2004 mean)
  - Minimum 3 years / 20 observations per well-gage pair (matching config)
  - Plot per-gage 2×2 scatter (one panel per quarter)

Outputs (all under results/quarterly_delta/):
  features/
    data_quarterly_deltas.csv
    regression_quarterly_by_well.csv
  figures/
    scatter_by_gage/      — one PNG per gage (2×2 quarter subplots)
    top10_by_r2/          — top-10 R² per gage per quarter (subplots)
    top10_by_r2_fit/      — top-10 combined scatter + per-well fit lines
    tables/
      top10_quarterly_by_r2.csv
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
OUT_DIR = RESULTS / "quarterly_delta"

FEAT_DIR = OUT_DIR / "features"
FIG_DIR  = OUT_DIR / "figures"
SCAT_DIR = FIG_DIR / "scatter_by_gage"
T10_DIR  = FIG_DIR / "top10_by_r2"
T10F_DIR = FIG_DIR / "top10_by_r2_fit"
TAB_DIR  = FIG_DIR / "tables"
for d in [FEAT_DIR, FIG_DIR, SCAT_DIR, T10_DIR, T10F_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MIN_OBS   = 20   # minimum quarter-year pairs per well-gage
MIN_YEARS = 3    # minimum years of data
MIN_FIT   = 5
TOP_N     = 10
CMAP      = cm.get_cmap("tab10")

QUARTER_MAP = {1: "Q1", 2: "Q1", 3: "Q1",
               4: "Q2", 5: "Q2", 6: "Q2",
               7: "Q3", 8: "Q3", 9: "Q3",
               10: "Q4", 11: "Q4", 12: "Q4"}
QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]
QUARTER_LABEL = {"Q1": "Q1 (Jan–Mar)", "Q2": "Q2 (Apr–Jun)",
                 "Q3": "Q3 (Jul–Sep)", "Q4": "Q4 (Oct–Dec)"}
QUARTER_COLOR = {"Q1": "#4C72B0", "Q2": "#55A868",
                 "Q3": "#E24A33", "Q4": "#C4AD3A"}

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

data["month"]   = data["date"].dt.month
data["quarter"] = data["month"].map(QUARTER_MAP)
data["qtr_year"] = data["date"].dt.year   # no cross-year adjustment needed

# ── Step 1: Quarterly means per well-gage-quarter-year ───────────────────────
print("Computing quarterly means...")
qtr_mean = (data.groupby(["well_id","gage_id","gage_name","quarter","qtr_year"])
            .agg(wte_qtr  =("wte","mean"),
                 q_qtr    =("q",  "mean"),
                 n_months =("wte","count"))
            .reset_index()
            .sort_values(["well_id","gage_id","quarter","qtr_year"]))

# Require all 3 months present in quarter
qtr_mean = qtr_mean[qtr_mean["n_months"] >= 3].copy()

# ── Step 2: Year-over-year delta within each quarter ─────────────────────────
qtr_mean["delta_wte_qtr"] = qtr_mean.groupby(
    ["well_id","gage_id","quarter"])["wte_qtr"].diff()
qtr_mean["delta_q_qtr"] = qtr_mean.groupby(
    ["well_id","gage_id","quarter"])["q_qtr"].diff()
qtr_mean["year_diff"] = qtr_mean.groupby(
    ["well_id","gage_id","quarter"])["qtr_year"].diff()

# Keep consecutive years only
qtr_mean = qtr_mean[qtr_mean["year_diff"] == 1].dropna(
    subset=["delta_wte_qtr","delta_q_qtr"])

# ── Outlier removal per gage (3-std on both delta_wte and delta_q) ───────────
def std_mask(series, k=3.0):
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        return pd.Series(True, index=series.index)
    return (series - mu).abs() <= k * sigma

before = len(qtr_mean)
mask_wte = qtr_mean.groupby(["gage_id"])["delta_wte_qtr"].transform(std_mask)
qtr_mean = qtr_mean[mask_wte].copy()
print(f"  Outlier removal (3-std on delta_wte per gage): {before - len(qtr_mean)} rows removed")

qtr_mean.to_csv(FEAT_DIR / "data_quarterly_deltas.csv", index=False)
print(f"  Saved data_quarterly_deltas.csv  ({len(qtr_mean)} rows)")

# ── Step 3: Per-well-quarter regression ──────────────────────────────────────
print("Running per-well regressions...")
reg_rows = []
for (wid, gid, gname, quarter), grp in qtr_mean.groupby(
        ["well_id","gage_id","gage_name","quarter"]):
    x = grp["delta_wte_qtr"].values
    y = grp["delta_q_qtr"].values
    n_years = grp["qtr_year"].nunique()
    if len(x) < MIN_OBS or n_years < MIN_YEARS or x.std() == 0:
        continue
    s, intercept, r_val, p_val, _ = linregress(x, y)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    reg_rows.append({
        "well_id": wid, "gage_id": gid, "gage_name": gname, "quarter": quarter,
        "n_obs": len(x), "n_years": n_years,
        "slope": round(s,4), "intercept": round(intercept,4),
        "r_squared": round(r_val**2,4), "r_value": round(r_val,4),
        "p_value": round(p_val,6),
        "pearson_r": round(pr,4), "pearson_p": round(pp,6),
        "spearman_r": round(sr,4), "spearman_p": round(sp,6),
    })

reg_df = pd.DataFrame(reg_rows)
reg_df.to_csv(FEAT_DIR / "regression_quarterly_by_well.csv", index=False)
print(f"  Saved regression_quarterly_by_well.csv  ({len(reg_df)} rows)")

# Summary
print("\n=== Quarterly regression summary ===")
print(f"{'Gage':<20} {'Qtr':<4} {'Wells':>6} {'%Neg':>6} "
      f"{'%Sig':>6} {'Median slope':>13} {'Median R²':>10}")
print("-"*72)
for (gname, quarter), grp in reg_df.groupby(["gage_name","quarter"]):
    short = GAGE_SHORT.get(gname, gname)[:18]
    pneg = 100*(grp["slope"]<0).mean()
    psig = 100*(grp["p_value"]<0.05).mean()
    print(f"{short:<20} {quarter:<4} {len(grp):>6} {pneg:>5.0f}% "
          f"{psig:>5.0f}% {grp['slope'].median():>13.3f} "
          f"{grp['r_squared'].median():>10.3f}")

# ── Step 4: Fig A — per-gage single scatter (all quarters combined, colour by quarter) ──
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
    gdata  = qtr_mean[qtr_mean["gage_name"] == gname].dropna(
                subset=["delta_wte_qtr","delta_q_qtr"])
    if gdata.empty:
        continue

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"{gname}\nΔQ vs ΔWTE — Same-quarter YoY delta (all wells, all quarters)",
                 fontsize=11, fontweight="bold")

    # scatter: colour by quarter, one point per well-year-quarter
    for quarter in QUARTER_ORDER:
        sub = gdata[gdata["quarter"] == quarter]
        if sub.empty:
            continue
        ax.scatter(sub["delta_wte_qtr"], sub["delta_q_qtr"],
                   s=16, color=QUARTER_COLOR[quarter], alpha=0.45,
                   edgecolors="none", zorder=3,
                   label=QUARTER_LABEL[quarter])

    # pooled fit across all quarters
    st = pooled_stats(gdata, "delta_wte_qtr", "delta_q_qtr")
    if st:
        xv = gdata["delta_wte_qtr"].values
        x_fit = np.linspace(xv.min(), xv.max(), 300)
        ax.plot(x_fit, st["slope"]*x_fit + st["intercept"],
                color="black", linewidth=2.0, linestyle="--",
                zorder=5, label="Pooled OLS")
        p_str = f"{st['p']:.4f}" if st['p'] >= 0.0001 else "<0.0001"
        ax.text(0.98, 0.97,
                f"All quarters pooled\nN={st['n']}  Wells={st['n_wells']}\n"
                f"Slope={st['slope']:.3f}\nR²={st['r2']:.3f}\np={p_str}",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white",
                          alpha=0.90, edgecolor="#AAAAAA"))

    ax.axhline(0, color="#CCCCCC", linewidth=0.7, linestyle=":")
    ax.axvline(0, color="#CCCCCC", linewidth=0.7, linestyle=":")
    ax.set_xlabel("ΔWTE same-quarter YoY (ft)", fontsize=10)
    ax.set_ylabel("ΔQ same-quarter YoY (cfs)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    out = SCAT_DIR / f"{gname.split()[0].lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

# ── Step 5: Fig B — top-10 R² per-well subplots, one fig per gage per quarter ─
print("\nGenerating top-10 per-well figures...")

for (gname, quarter), grp in reg_df.groupby(["gage_name","quarter"]):
    gshort = GAGE_SHORT.get(gname, gname)
    ranked = grp.nlargest(TOP_N, "r_squared")
    top_ids = ranked["well_id"].tolist()
    mvals   = dict(zip(ranked["well_id"].astype(str), ranked["r_squared"]))

    sub = qtr_mean[(qtr_mean["gage_name"]==gname) &
                   (qtr_mean["quarter"]==quarter) &
                   (qtr_mean["well_id"].isin(top_ids))
                   ].dropna(subset=["delta_wte_qtr","delta_q_qtr"])
    present = [w for w in top_ids if w in set(sub["well_id"].unique())]
    if not present:
        continue

    n = len(present)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.2*ncols, 4.0*nrows), squeeze=False)
    axes_flat = axes.flatten()

    all_x = sub["delta_wte_qtr"].values
    all_y = sub["delta_q_qtr"].values
    xpad  = (all_x.max()-all_x.min())*0.05 or 1
    ypad  = (all_y.max()-all_y.min())*0.05 or 1
    xlim  = (all_x.min()-xpad, all_x.max()+xpad)
    ylim  = (all_y.min()-ypad, all_y.max()+ypad)

    for i, wid in enumerate(present):
        ax = axes_flat[i]
        color = CMAP(i % 10)
        wd = sub[sub["well_id"]==wid]
        x, y = wd["delta_wte_qtr"].values, wd["delta_q_qtr"].values
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

    fig.suptitle(f"{gname} — {QUARTER_LABEL[quarter]}\n"
                 f"ΔQ vs ΔWTE per well (top {n} by R², quarterly delta)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = T10_DIR / f"{gname.split()[0].lower()}_{quarter.lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

# ── Step 6: Fig C — top-10 combined scatter + per-well fit lines ─────────────
print("\nGenerating top-10 combined fit figures...")

for (gname, quarter), grp in reg_df.groupby(["gage_name","quarter"]):
    gshort = GAGE_SHORT.get(gname, gname)
    ranked = grp.nlargest(TOP_N, "r_squared")
    top_ids = ranked["well_id"].tolist()

    sub = qtr_mean[(qtr_mean["gage_name"]==gname) &
                   (qtr_mean["quarter"]==quarter) &
                   (qtr_mean["well_id"].isin(top_ids))
                   ].dropna(subset=["delta_wte_qtr","delta_q_qtr"])
    wells = sub["well_id"].unique()
    if len(wells) == 0:
        continue

    fig, ax = plt.subplots(figsize=(9, 6))
    all_x = sub["delta_wte_qtr"].values
    all_y = sub["delta_q_qtr"].values
    xpad = (all_x.max()-all_x.min())*0.05 or 1
    xlim = (all_x.min()-xpad, all_x.max()+xpad)
    x_fit = np.linspace(xlim[0], xlim[1], 200)

    legend_entries = []
    for i, wid in enumerate(wells):
        color = CMAP(i % 10)
        wd = sub[sub["well_id"]==wid]
        x, y = wd["delta_wte_qtr"].values, wd["delta_q_qtr"].values
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
    ax.set_xlabel("ΔWTE quarterly (ft/yr)", fontsize=11)
    ax.set_ylabel("ΔQ quarterly (cfs/yr)", fontsize=11)
    ax.set_title(f"{gname} — {QUARTER_LABEL[quarter]}\n"
                 f"ΔQ vs ΔWTE — Quarterly delta (top {len(wells)} by R², per-well fit)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = T10F_DIR / f"{gname.split()[0].lower()}_{quarter.lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
tables = []
for (gname, quarter), grp in reg_df.groupby(["gage_name","quarter"]):
    ranked = grp.nlargest(TOP_N, "r_squared").copy()
    ranked.insert(0, "rank", range(1, len(ranked)+1))
    tables.append(ranked)
top10 = pd.concat(tables).reset_index(drop=True)
top10.to_csv(TAB_DIR / "top10_quarterly_by_r2.csv", index=False)
print(f"\n  Saved → {TAB_DIR / 'top10_quarterly_by_r2.csv'}")
print(f"\nAll outputs → {OUT_DIR}")
