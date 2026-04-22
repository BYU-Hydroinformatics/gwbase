"""
基流分离后的 delta 回归（Eckhardt数字滤波器）

文献背景：
  Eckhardt (2005, Hydrological Processes) 提出的两参数递推数字滤波器是基流分离
  最广泛使用的方法之一，大量USGS报告和近年流域研究（如Bieger et al. 2017）采用。
  基流代表地下水对径流的直接贡献，理论上与WTE的关系应比总流量更直接。

方法：
  1. 对每个 gage 的月均 Q 应用 Eckhardt 滤波器，分离出月均基流（BF）
     BF_t = [(1-BFImax)*α*BF_{t-1} + (1-α)*BFImax*Q_t] / (1-α*BFImax)
     约束：BF_t ≤ Q_t
     参数：α=0.98（退水常数），BFImax=0.80（最大基流指数，多年平均流域）
  2. 计算各 well-gage 对的年均 WTE 和年均基流
  3. 年际差分：ΔBFQ_ann 和 ΔWTE_ann
  4. 回归 ΔBFQ ~ ΔWTE

结果目录：results/baseflow_delta/
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
OUT_DIR = RESULTS / "baseflow_delta"

FEAT_DIR = OUT_DIR / "features"
FIG_DIR  = OUT_DIR / "figures"
SCAT_DIR = FIG_DIR / "scatter_by_gage"
T10_DIR  = FIG_DIR / "top10_by_r2"
T10F_DIR = FIG_DIR / "top10_by_r2_fit"
TAB_DIR  = FIG_DIR / "tables"
for d in [FEAT_DIR, FIG_DIR, SCAT_DIR, T10_DIR, T10F_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Eckhardt filter parameters
ALPHA   = 0.98   # 月度退水常数（日尺度通常0.998，月度适当降低）
BFIMAX  = 0.80   # 最大基流指数（perennial streams，多年平均）

MIN_OBS  = 8
MIN_YEARS= 3
MIN_FIT  = 5
TOP_N    = 10
CMAP     = cm.get_cmap("tab10")

GAGE_SHORT = {
    "BEAR RIVER NEAR CORINNE - UT":                  "Bear River",
    "WEBER RIVER NEAR PLAIN CITY - UT":               "Weber River",
    "PROVO RIVER AT PROVO - UT":                      "Provo River",
    "SPANISH FORK NEAR LAKE SHORE - UTAH":            "Spanish Fork",
    "LITTLE COTTONWOOD CREEK @ JORDAN RIVER NR SLC":  "Little Cottonwood",
}

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading data...")
data = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv",
                   parse_dates=["date"])
data["well_id"] = data["well_id"].astype(str)
data["gage_id"] = data["gage_id"].astype(str)
data["year"]    = data["date"].dt.year
data["month"]   = data["date"].dt.month

# ── Step 1: Eckhardt 基流分离（按 gage_id 分组，在月均 Q 上运行）────────────────
print("Applying Eckhardt baseflow filter per gage...")

def eckhardt_filter(q_series, alpha=ALPHA, bfimax=BFIMAX):
    """
    Eckhardt (2005) 两参数递推数字滤波器。
    q_series: 按时间排序的月均流量序列（无缺失）。
    返回：等长的基流序列。
    """
    denom = 1 - alpha * bfimax
    bf = np.empty(len(q_series))
    bf[0] = q_series.iloc[0] * bfimax   # 初始值
    for t in range(1, len(q_series)):
        bf[t] = ((1 - bfimax) * alpha * bf[t-1]
                 + (1 - alpha) * bfimax * q_series.iloc[t]) / denom
        bf[t] = min(bf[t], q_series.iloc[t])   # 基流不超过总流量
    return bf

# 对每个 gage 构建完整月度 Q 序列，应用滤波器
gage_monthly = (data.groupby(["gage_id","gage_name","year","month"])
                .agg(q_mon=("q","mean"))
                .reset_index()
                .sort_values(["gage_id","year","month"]))

bf_rows = []
for (gid, gname), grp in gage_monthly.groupby(["gage_id","gage_name"]):
    grp = grp.sort_values(["year","month"]).reset_index(drop=True)
    # 补全缺失月份（用线性插值）
    full_idx = pd.MultiIndex.from_product(
        [range(grp["year"].min(), grp["year"].max()+1), range(1,13)],
        names=["year","month"])
    grp_full = (grp.set_index(["year","month"])["q_mon"]
                .reindex(full_idx).interpolate(limit=3).fillna(method="ffill").fillna(method="bfill"))
    if grp_full.isna().any():
        continue
    bf_vals = eckhardt_filter(grp_full)
    for (yr, mo), q_val, bf_val in zip(grp_full.index, grp_full.values, bf_vals):
        bf_rows.append({"gage_id":gid,"gage_name":gname,
                        "year":yr,"month":mo,"q_mon":q_val,"bf_mon":bf_val})

bf_df = pd.DataFrame(bf_rows)
print(f"  Baseflow computed for {bf_df['gage_id'].nunique()} gages")

# ── Step 2: 合并 well-gage 对，计算年均 WTE 和年均基流 ────────────────────────
# well 侧：年均 WTE（≥6个月）
wte_ann = (data.groupby(["well_id","gage_id","gage_name","year"])
           .agg(wte_ann=("wte","mean"), n_wte=("wte","count"))
           .reset_index())
wte_ann = wte_ann[wte_ann["n_wte"]>=6].copy()

# gage 侧：年均基流（≥6个月）
bf_ann = (bf_df.groupby(["gage_id","year"])
          .agg(bf_ann=("bf_mon","mean"), n_bf=("bf_mon","count"))
          .reset_index())
bf_ann = bf_ann[bf_ann["n_bf"]>=6].copy()

paired = wte_ann.merge(bf_ann, on=["gage_id","year"])
paired = paired.sort_values(["well_id","gage_id","year"])

# ── Step 3: 年际差分 ──────────────────────────────────────────────────────────
paired["delta_wte"] = paired.groupby(["well_id","gage_id"])["wte_ann"].diff()
paired["delta_bf"]  = paired.groupby(["well_id","gage_id"])["bf_ann"].diff()
paired["year_diff"] = paired.groupby(["well_id","gage_id"])["year"].diff()

paired = paired[paired["year_diff"]==1].dropna(subset=["delta_wte","delta_bf"])

# Outlier removal
def iqr_mask(series, k=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0: return pd.Series(True, index=series.index)
    return (series >= q1-k*iqr) & (series <= q3+k*iqr)

before = len(paired)
paired = paired[paired.groupby(["well_id","gage_id"])["delta_wte"]
                .transform(iqr_mask)].copy()
print(f"  Outlier removal: {before-len(paired)} rows removed")
paired.to_csv(FEAT_DIR / "data_baseflow_deltas.csv", index=False)
print(f"  Saved ({len(paired)} rows)")

# ── Step 4: Per-well regression ───────────────────────────────────────────────
print("Running per-well regressions...")
reg_rows = []
for (wid, gid, gname), grp in paired.groupby(["well_id","gage_id","gage_name"]):
    x, y = grp["delta_wte"].values, grp["delta_bf"].values
    n_years = grp["year"].nunique()
    if len(x)<MIN_OBS or n_years<MIN_YEARS or x.std()==0: continue
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
reg_df.to_csv(FEAT_DIR / "regression_baseflow_by_well.csv", index=False)
print(f"  Saved regression ({len(reg_df)} rows)")

print("\n=== Baseflow separation Δ — regression summary ===")
print(f"{'Gage':<22} {'Wells':>6} {'%Neg':>6} {'%Sig':>6} "
      f"{'Median slope':>13} {'Median R²':>10}")
print("-"*70)
for gname, grp in reg_df.groupby("gage_name"):
    short = GAGE_SHORT.get(gname,gname)[:20]
    print(f"{short:<22} {len(grp):>6} "
          f"{100*(grp['slope']<0).mean():>5.0f}% "
          f"{100*(grp['p_value']<0.05).mean():>5.0f}% "
          f"{grp['slope'].median():>13.3f} "
          f"{grp['r_squared'].median():>10.3f}")

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
    gdata = paired[paired["gage_name"]==gname]
    if gdata.empty: continue
    fig, ax = plt.subplots(figsize=(9,6))
    wells = gdata["well_id"].unique()
    for i, wid in enumerate(wells):
        wd = gdata[gdata["well_id"]==wid].dropna(subset=["delta_wte","delta_bf"])
        ax.scatter(wd["delta_wte"], wd["delta_bf"],
                   s=20, color=CMAP(i%10), alpha=0.55,
                   edgecolors="none", zorder=3)
    st = pooled_stats(gdata, "delta_wte", "delta_bf")
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
    ax.set_xlabel("ΔWTE annual (ft/yr)", fontsize=11)
    ax.set_ylabel("ΔBaseflow annual (cfs/yr)", fontsize=11)
    ax.set_title(f"{gname}\nΔBaseflow vs ΔWTE — Eckhardt filter (α={ALPHA}, BFImax={BFIMAX})",
                 fontsize=10, fontweight="bold")
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
    sub = paired[(paired["gage_name"]==gname)&(paired["well_id"].isin(top_ids))
                 ].dropna(subset=["delta_wte","delta_bf"])
    present = [w for w in top_ids if w in set(sub["well_id"].unique())]
    if not present: continue
    n = len(present)
    ncols = min(5,n); nrows = math.ceil(n/ncols)
    fig, axes = plt.subplots(nrows,ncols,figsize=(4.2*ncols,4.0*nrows),squeeze=False)
    axes_flat = axes.flatten()
    all_x = sub["delta_wte"].values; all_y = sub["delta_bf"].values
    xpad = (all_x.max()-all_x.min())*0.05 or 1
    ypad = (all_y.max()-all_y.min())*0.05 or 1
    xlim = (all_x.min()-xpad, all_x.max()+xpad)
    ylim = (all_y.min()-ypad, all_y.max()+ypad)
    for i, wid in enumerate(present):
        ax = axes_flat[i]
        wd = sub[sub["well_id"]==wid]
        x, y = wd["delta_wte"].values, wd["delta_bf"].values
        ax.scatter(x,y,s=28,color=CMAP(i%10),alpha=0.8,edgecolors="none",zorder=3)
        stats = [f"n={len(x)}yr"]
        if len(x)>=MIN_FIT and x.std()>0:
            s,ic,r_val,p_val,_ = linregress(x,y)
            x_fit = np.linspace(xlim[0],xlim[1],200)
            ax.plot(x_fit,s*x_fit+ic,color="red",linewidth=1.4,zorder=4)
            p_str = f"{p_val:.3e}" if p_val<0.001 else f"{p_val:.3f}"
            stats += [f"slope={s:.3f}",f"R²={r_val**2:.3f}",f"p={p_str}"]
        ax.text(0.97,0.97,"\n".join(stats),transform=ax.transAxes,fontsize=7.5,
                ha="right",va="top",
                bbox=dict(boxstyle="round,pad=0.25",facecolor="white",alpha=0.85,edgecolor="#AAAAAA"))
        ax.axhline(0,color="#CCCCCC",linewidth=0.6,linestyle=":")
        ax.axvline(0,color="#CCCCCC",linewidth=0.6,linestyle=":")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("ΔWTE (ft/yr)",fontsize=8)
        ax.set_ylabel("ΔBF (cfs/yr)",fontsize=8)
        mv = mvals.get(str(wid))
        ax.set_title(f"#{i+1}  {wid}"+(f"  R²={mv:.3f}" if mv else ""),fontsize=7.5,fontweight="bold")
        ax.grid(True,alpha=0.2)
    for j in range(n,len(axes_flat)): axes_flat[j].set_visible(False)
    fig.suptitle(f"{gname}\nΔBaseflow vs ΔWTE per well (top {n} by R², Eckhardt filter)",
                 fontsize=10,fontweight="bold")
    plt.tight_layout()
    out = T10_DIR / f"{gname.split()[0].lower()}.png"
    plt.savefig(out,dpi=160,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

print("\nGenerating top-10 combined fit figures...")
for gname, grp in reg_df.groupby("gage_name"):
    ranked = grp.nlargest(TOP_N,"r_squared")
    top_ids = ranked["well_id"].tolist()
    sub = paired[(paired["gage_name"]==gname)&(paired["well_id"].isin(top_ids))
                 ].dropna(subset=["delta_wte","delta_bf"])
    wells = sub["well_id"].unique()
    if not len(wells): continue
    fig,ax = plt.subplots(figsize=(9,6))
    all_x = sub["delta_wte"].values; all_y = sub["delta_bf"].values
    xpad = (all_x.max()-all_x.min())*0.05 or 1
    xlim = (all_x.min()-xpad, all_x.max()+xpad)
    x_fit = np.linspace(xlim[0],xlim[1],200)
    legend_entries = []
    for i, wid in enumerate(wells):
        color = CMAP(i%10)
        wd = sub[sub["well_id"]==wid]
        x, y = wd["delta_wte"].values, wd["delta_bf"].values
        sc = ax.scatter(x,y,s=22,color=color,alpha=0.65,edgecolors="none",zorder=3)
        label = str(wid)
        if len(x)>=MIN_FIT and x.std()>0:
            s,ic,r_val,p_val,_ = linregress(x,y)
            ax.plot(x_fit,s*x_fit+ic,color=color,linewidth=1.3,alpha=0.9,zorder=4)
            label += f"  (s={s:.2f}, R²={r_val**2:.3f})"
        legend_entries.append((sc,label))
    s_all,ic_all,r_all,p_all,_ = linregress(all_x,all_y)
    ax.plot(x_fit,s_all*x_fit+ic_all,color="black",linewidth=2,linestyle="--",zorder=6)
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
              title=f"Top {len(wells)} by R²",title_fontsize=7.5)
    ax.set_xlabel("ΔWTE annual (ft/yr)",fontsize=11)
    ax.set_ylabel("ΔBaseflow annual (cfs/yr)",fontsize=11)
    ax.set_title(f"{gname}\nΔBaseflow vs ΔWTE (top {len(wells)}, per-well fit)",
                 fontsize=10,fontweight="bold")
    ax.grid(True,alpha=0.25)
    plt.tight_layout()
    out = T10F_DIR / f"{gname.split()[0].lower()}.png"
    plt.savefig(out,dpi=160,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

tables = []
for gname,grp in reg_df.groupby("gage_name"):
    ranked = grp.nlargest(TOP_N,"r_squared").copy()
    ranked.insert(0,"rank",range(1,len(ranked)+1))
    tables.append(ranked)
pd.concat(tables).reset_index(drop=True).to_csv(TAB_DIR/"top10_baseflow_by_r2.csv",index=False)
print(f"\nAll outputs → {OUT_DIR}")
