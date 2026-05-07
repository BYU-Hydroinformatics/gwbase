"""
12个月滚动均值年际差分：先平滑掉季节波动，再做年际差分。

逻辑：
  1. 对每个 well-gage 的月度 WTE 和 Q 计算 12 个月滚动均值
     （至少需要 10 个月有效数据，min_periods=10）
  2. 对滚动均值做 12 个月差分：rolling_mean_t - rolling_mean_{t-12}
     即：当前 12m 均值 vs 一年前的 12m 均值
  3. 12m 滚动均值本身已完全消除季节性，差分反映纯年际变化趋势

结果目录：results/rolling12m_delta/
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
OUT_DIR = RESULTS / "rolling12m_delta"

FEAT_DIR = OUT_DIR / "features"
FIG_DIR  = OUT_DIR / "figures"
SCAT_DIR = FIG_DIR / "scatter_by_gage"
T10_DIR  = FIG_DIR / "top10_by_r2"
T10F_DIR = FIG_DIR / "top10_by_r2_fit"
TAB_DIR  = FIG_DIR / "tables"
for d in [FEAT_DIR, FIG_DIR, SCAT_DIR, T10_DIR, T10F_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MIN_OBS   = 8    # 年际差分，数据量少，门槛降低
MIN_YEARS = 3
MIN_FIT   = 5
TOP_N     = 10
CMAP      = cm.get_cmap("tab10")

GAGE_SHORT = {
    "BEAR RIVER NEAR CORINNE - UT":                  "Bear River",
    "WEBER RIVER NEAR PLAIN CITY - UT":               "Weber River",
    "PROVO RIVER AT PROVO - UT":                      "Provo River",
    "SPANISH FORK NEAR LAKE SHORE - UTAH":            "Spanish Fork",
    "LITTLE COTTONWOOD CREEK @ JORDAN RIVER NR SLC":  "Little Cottonwood",
}

QUARTER_MAP = {1:"Q1",2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",
               7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"}
QUARTER_COLOR = {"Q1":"#4C72B0","Q2":"#55A868","Q3":"#E24A33","Q4":"#C4AD3A"}

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading data...")
data = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv",
                   parse_dates=["date"])
data["well_id"] = data["well_id"].astype(str)
data["gage_id"] = data["gage_id"].astype(str)
data["month"]   = data["date"].dt.month
data["year"]    = data["date"].dt.year
data["quarter"] = data["month"].map(QUARTER_MAP)
data["month_idx"] = data["year"] * 12 + data["month"]
data = data.sort_values(["well_id","gage_id","month_idx"]).reset_index(drop=True)

# ── Step 1: 12 个月滚动均值 ───────────────────────────────────────────────────
print("Computing 12-month rolling means...")
data["wte_roll12"] = (data.groupby(["well_id","gage_id"])["wte"]
                      .transform(lambda x: x.rolling(12, min_periods=10).mean()))
data["q_roll12"]   = (data.groupby(["well_id","gage_id"])["q"]
                      .transform(lambda x: x.rolling(12, min_periods=10).mean()))

# ── Step 2: 12 个月差分（年际变化） ───────────────────────────────────────────
print("Computing 12-month diff on rolling means...")
data["delta_wte"]   = (data.groupby(["well_id","gage_id"])["wte_roll12"]
                       .transform(lambda x: x.diff(12)))
data["delta_q"]     = (data.groupby(["well_id","gage_id"])["q_roll12"]
                       .transform(lambda x: x.diff(12)))
data["month_diff"]  = data.groupby(["well_id","gage_id"])["month_idx"].diff(12)

# 只保留恰好跨 12 个月的有效行
data = data[(data["month_diff"] == 12)].dropna(subset=["delta_wte","delta_q"])

# ── Outlier removal (IQR×3) ──────────────────────────────────────────────────
def iqr_mask(series, k=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(True, index=series.index)
    return (series >= q1 - k*iqr) & (series <= q3 + k*iqr)

before = len(data)
data = data[data.groupby(["well_id","gage_id"])["delta_wte"].transform(iqr_mask)].copy()
print(f"  Outlier removal: {before - len(data)} rows removed")

data.to_csv(FEAT_DIR / "data_rolling12m_deltas.csv", index=False)
print(f"  Saved  ({len(data)} rows)")

# ── Step 3: Per-well regression ───────────────────────────────────────────────
print("Running per-well regressions...")
reg_rows = []
for (wid, gid, gname), grp in data.groupby(["well_id","gage_id","gage_name"]):
    x, y = grp["delta_wte"].values, grp["delta_q"].values
    n_years = grp["year"].nunique()
    if len(x) < MIN_OBS or n_years < MIN_YEARS or x.std() == 0:
        continue
    s, ic, r_val, p_val, _ = linregress(x, y)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    reg_rows.append({"well_id":wid,"gage_id":gid,"gage_name":gname,
                     "n_obs":len(x),"n_years":n_years,
                     "slope":round(s,4),"intercept":round(ic,4),
                     "r_squared":round(r_val**2,4),"r_value":round(r_val,4),
                     "p_value":round(p_val,6),
                     "pearson_r":round(pr,4),"pearson_p":round(pp,6),
                     "spearman_r":round(sr,4),"spearman_p":round(sp,6)})

reg_df = pd.DataFrame(reg_rows)
reg_df.to_csv(FEAT_DIR / "regression_rolling12m_by_well.csv", index=False)
print(f"  Saved regression  ({len(reg_df)} rows)")

print("\n=== 12-month rolling mean Δ — regression summary ===")
print(f"{'Gage':<22} {'Wells':>6} {'%Neg':>6} {'%Sig':>6} "
      f"{'Median slope':>13} {'Median R²':>10} {'Median N':>9}")
print("-"*78)
for gname, grp in reg_df.groupby("gage_name"):
    short = GAGE_SHORT.get(gname, gname)[:20]
    print(f"{short:<22} {len(grp):>6} "
          f"{100*(grp['slope']<0).mean():>5.0f}% "
          f"{100*(grp['p_value']<0.05).mean():>5.0f}% "
          f"{grp['slope'].median():>13.3f} "
          f"{grp['r_squared'].median():>10.3f} "
          f"{grp['n_obs'].median():>9.0f}")

# ── Figures ───────────────────────────────────────────────────────────────────
def pooled_stats(df, xcol, ycol):
    sub = df[[xcol,ycol]].dropna()
    x, y = sub[xcol].values, sub[ycol].values
    if len(x)<5 or x.std()==0: return None
    s, ic, r_val, p_val, _ = linregress(x, y)
    return {"slope":s,"r2":r_val**2,"p":p_val,
            "n":len(x),"n_wells":df["well_id"].nunique(),"intercept":ic}

print("\nGenerating scatter figures...")
for gname in GAGE_SHORT:
    gdata = data[data["gage_name"]==gname]
    if gdata.empty: continue
    fig, ax = plt.subplots(figsize=(9,6))
    for q, color in QUARTER_COLOR.items():
        sub = gdata[gdata["quarter"]==q].dropna(subset=["delta_wte","delta_q"])
        if not sub.empty:
            ax.scatter(sub["delta_wte"], sub["delta_q"],
                       s=14, color=color, alpha=0.45,
                       edgecolors="none", zorder=3, label=q)
    st = pooled_stats(gdata, "delta_wte", "delta_q")
    if st:
        xv = gdata["delta_wte"].dropna().values
        x_fit = np.linspace(xv.min(), xv.max(), 200)
        ax.plot(x_fit, st["slope"]*x_fit+st["intercept"],
                color="black", linewidth=2, linestyle="--", zorder=6)
        p_str = f"{st['p']:.4f}" if st['p']>=0.0001 else "<0.0001"
        ax.text(0.98,0.97,
                f"N={st['n']}  Wells={st['n_wells']}\n"
                f"Slope={st['slope']:.3f}\nR²={st['r2']:.3f}\np={p_str}",
                transform=ax.transAxes,fontsize=9,ha="right",va="top",
                bbox=dict(boxstyle="round",facecolor="white",alpha=0.88,edgecolor="#AAAAAA"))
    ax.axhline(0,color="#CCCCCC",linewidth=0.7,linestyle=":")
    ax.axvline(0,color="#CCCCCC",linewidth=0.7,linestyle=":")
    ax.set_xlabel("Δ(12m rolling WTE) ft/yr", fontsize=11)
    ax.set_ylabel("Δ(12m rolling Q) cfs/yr", fontsize=11)
    ax.set_title(f"{gname}\nΔQ vs ΔWTE — 12-month rolling mean, annual diff (all wells)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85,
              title="Quarter of current month")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = SCAT_DIR / f"{gname.split()[0].lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

print("\nGenerating top-10 per-well figures...")
for gname, grp in reg_df.groupby("gage_name"):
    ranked = grp.nlargest(TOP_N,"r_squared")
    top_ids = ranked["well_id"].tolist()
    mvals   = dict(zip(ranked["well_id"].astype(str), ranked["r_squared"]))
    sub = data[(data["gage_name"]==gname) & (data["well_id"].isin(top_ids))
               ].dropna(subset=["delta_wte","delta_q"])
    present = [w for w in top_ids if w in set(sub["well_id"].unique())]
    if not present: continue
    n = len(present)
    ncols = min(5,n); nrows = math.ceil(n/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols,4.0*nrows), squeeze=False)
    axes_flat = axes.flatten()
    all_x = sub["delta_wte"].values; all_y = sub["delta_q"].values
    xpad = (all_x.max()-all_x.min())*0.05 or 1
    ypad = (all_y.max()-all_y.min())*0.05 or 1
    xlim = (all_x.min()-xpad, all_x.max()+xpad)
    ylim = (all_y.min()-ypad, all_y.max()+ypad)
    for i, wid in enumerate(present):
        ax = axes_flat[i]
        wd = sub[sub["well_id"]==wid]
        for q, color in QUARTER_COLOR.items():
            qd = wd[wd["quarter"]==q]
            if not qd.empty:
                ax.scatter(qd["delta_wte"], qd["delta_q"],
                           s=18, color=color, alpha=0.75, edgecolors="none", zorder=3)
        x, y = wd["delta_wte"].values, wd["delta_q"].values
        stats = [f"n={len(x)}"]
        if len(x)>=MIN_FIT and x.std()>0:
            s, ic, r_val, p_val, _ = linregress(x, y)
            x_fit = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(x_fit, s*x_fit+ic, color="black", linewidth=1.4,
                    linestyle="--", zorder=4)
            p_str = f"{p_val:.3e}" if p_val<0.001 else f"{p_val:.3f}"
            stats += [f"slope={s:.3f}", f"R²={r_val**2:.3f}", f"p={p_str}"]
        ax.text(0.97,0.97,"\n".join(stats),transform=ax.transAxes,fontsize=7.5,
                ha="right",va="top",
                bbox=dict(boxstyle="round,pad=0.25",facecolor="white",
                          alpha=0.85,edgecolor="#AAAAAA"))
        ax.axhline(0,color="#CCCCCC",linewidth=0.6,linestyle=":")
        ax.axvline(0,color="#CCCCCC",linewidth=0.6,linestyle=":")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("ΔWTE roll12 (ft)", fontsize=8)
        ax.set_ylabel("ΔQ roll12 (cfs)", fontsize=8)
        mv = mvals.get(str(wid))
        ax.set_title(f"#{i+1}  {wid}"+(f"  R²={mv:.3f}" if mv else ""),
                     fontsize=7.5, fontweight="bold")
        ax.grid(True, alpha=0.2)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"{gname}\nΔQ vs ΔWTE per well (top {n} by R², 12m rolling Δ)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = T10_DIR / f"{gname.split()[0].lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

print("\nGenerating top-10 combined fit figures...")
for gname, grp in reg_df.groupby("gage_name"):
    ranked = grp.nlargest(TOP_N,"r_squared")
    top_ids = ranked["well_id"].tolist()
    sub = data[(data["gage_name"]==gname) & (data["well_id"].isin(top_ids))
               ].dropna(subset=["delta_wte","delta_q"])
    wells = sub["well_id"].unique()
    if not len(wells): continue
    fig, ax = plt.subplots(figsize=(9,6))
    all_x = sub["delta_wte"].values; all_y = sub["delta_q"].values
    xpad = (all_x.max()-all_x.min())*0.05 or 1
    xlim = (all_x.min()-xpad, all_x.max()+xpad)
    x_fit = np.linspace(xlim[0], xlim[1], 200)
    legend_entries = []
    for i, wid in enumerate(wells):
        color = CMAP(i%10)
        wd = sub[sub["well_id"]==wid]
        x, y = wd["delta_wte"].values, wd["delta_q"].values
        sc = ax.scatter(x, y, s=18, color=color, alpha=0.65, edgecolors="none", zorder=3)
        label = str(wid)
        if len(x)>=MIN_FIT and x.std()>0:
            s, ic, r_val, p_val, _ = linregress(x, y)
            ax.plot(x_fit, s*x_fit+ic, color=color, linewidth=1.3, alpha=0.9, zorder=4)
            label += f"  (s={s:.2f}, R²={r_val**2:.3f})"
        legend_entries.append((sc, label))
    s_all, ic_all, r_all, p_all, _ = linregress(all_x, all_y)
    ax.plot(x_fit, s_all*x_fit+ic_all, color="black", linewidth=2,
            linestyle="--", zorder=6)
    ax.axhline(0,color="#CCCCCC",linewidth=0.7,linestyle=":")
    ax.axvline(0,color="#CCCCCC",linewidth=0.7,linestyle=":")
    p_str = f"{p_all:.4f}" if p_all>=0.0001 else "<0.0001"
    ax.text(0.98,0.97,
            f"Overall\nN={len(all_x)}  Wells={len(wells)}\n"
            f"Slope={s_all:.3f}\nR²={r_all**2:.3f}\np={p_str}",
            transform=ax.transAxes,fontsize=8.5,ha="right",va="top",
            bbox=dict(boxstyle="round",facecolor="white",alpha=0.88,edgecolor="#AAAAAA"))
    ax.legend([e[0] for e in legend_entries],[e[1] for e in legend_entries],
              fontsize=7,loc="lower right",framealpha=0.85,
              title=f"Top {len(wells)} wells by R²",title_fontsize=7.5)
    ax.set_xlabel("Δ(12m rolling WTE) ft/yr", fontsize=11)
    ax.set_ylabel("Δ(12m rolling Q) cfs/yr", fontsize=11)
    ax.set_title(f"{gname}\n12m rolling Δ (top {len(wells)}, per-well fit)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = T10F_DIR / f"{gname.split()[0].lower()}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

tables = []
for gname, grp in reg_df.groupby("gage_name"):
    ranked = grp.nlargest(TOP_N,"r_squared").copy()
    ranked.insert(0,"rank",range(1,len(ranked)+1))
    tables.append(ranked)
pd.concat(tables).reset_index(drop=True).to_csv(
    TAB_DIR / "top10_rolling12m_by_r2.csv", index=False)
print(f"\nAll outputs → {OUT_DIR}")
