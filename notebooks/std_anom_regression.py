"""
标准化距平直接回归（Standardized Anomaly Regression）

文献背景：
  标准化距平（Z-score anomaly）是气候和水文学中去除季节性最直接的方法
  （参考：IPCC标准流程、Kuss & Gurdak 2014 JHYDROL、Tremblay et al. 2011）。
  不做差分，而是直接问：当某月WTE比该月历史均值高X个标准差时，
  Q是否也同步偏高？斜率=1表示完全协同变化，斜率>0表示地下水补给河流。

方法：
  1. 计算各 well-gage 对每个自然月（1–12月）的多年均值和标准差
  2. 标准化：z_wte = (wte - mean_m) / std_m，z_q 同理
  3. 直接回归 z_q ~ z_wte（无差分）
     注：先移除线性长期趋势再标准化，避免趋势影响
  4. 每个数据点 = 一个月度观测，x轴/y轴单位均为"标准差"

结果目录：results/std_anom_regression/
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
OUT_DIR = RESULTS / "std_anom_regression"

FEAT_DIR = OUT_DIR / "features"
FIG_DIR  = OUT_DIR / "figures"
SCAT_DIR = FIG_DIR / "scatter_by_gage"
T10_DIR  = FIG_DIR / "top10_by_r2"
T10F_DIR = FIG_DIR / "top10_by_r2_fit"
TAB_DIR  = FIG_DIR / "tables"
for d in [FEAT_DIR, FIG_DIR, SCAT_DIR, T10_DIR, T10F_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MIN_OBS   = 36   # 月度直接回归，数据量充足，要求更多样本
MIN_YEARS = 5
MIN_FIT   = 10
TOP_N     = 10
CMAP      = cm.get_cmap("tab10")

QUARTER_MAP = {1:"Q1",2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",
               7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"}
QUARTER_COLOR = {"Q1":"#4C72B0","Q2":"#55A868","Q3":"#E24A33","Q4":"#C4AD3A"}

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
data["month"]   = data["date"].dt.month
data["year"]    = data["date"].dt.year
data["quarter"] = data["month"].map(QUARTER_MAP)
data["month_idx"] = data["year"]*12 + data["month"]

# ── Step 1: 移除线性时间趋势（避免共同趋势驱动虚假相关）───────────────────────
print("Removing linear trends per well-gage pair...")

def detrend_series(series, time_idx):
    """OLS去线性趋势，返回残差。"""
    mask = ~(np.isnan(series) | np.isnan(time_idx))
    if mask.sum() < 12:
        return series
    s, ic, _, _, _ = linregress(time_idx[mask], series[mask])
    trend = s * time_idx + ic
    return series - trend

data["wte_dt"] = (data.groupby(["well_id","gage_id"])
                  .apply(lambda g: pd.Series(
                      detrend_series(g["wte"].values, g["month_idx"].values.astype(float)),
                      index=g.index))
                  .reset_index(level=[0,1], drop=True))

data["q_dt"]   = (data.groupby(["well_id","gage_id"])
                  .apply(lambda g: pd.Series(
                      detrend_series(g["q"].values, g["month_idx"].values.astype(float)),
                      index=g.index))
                  .reset_index(level=[0,1], drop=True))

# ── Step 2: 计算月度 climatology（均值和标准差），做标准化 ─────────────────────
print("Computing standardized anomalies...")
clim = (data.groupby(["well_id","gage_id","month"])
        .agg(wte_mean=("wte_dt","mean"), wte_std=("wte_dt","std"),
             q_mean  =("q_dt",  "mean"), q_std  =("q_dt",  "std"))
        .reset_index())

# 防止 std=0 的情况
clim["wte_std"] = clim["wte_std"].replace(0, np.nan)
clim["q_std"]   = clim["q_std"].replace(0, np.nan)

data = data.merge(clim, on=["well_id","gage_id","month"])
data["z_wte"] = (data["wte_dt"] - data["wte_mean"]) / data["wte_std"]
data["z_q"]   = (data["q_dt"]   - data["q_mean"])   / data["q_std"]

data = data.dropna(subset=["z_wte","z_q"])

# 截断极端值（>4σ）
data = data[(data["z_wte"].abs()<=4) & (data["z_q"].abs()<=4)].copy()

data.to_csv(FEAT_DIR / "data_std_anom.csv", index=False)
print(f"  Saved ({len(data)} rows)")

# ── Step 3: Per-well regression ───────────────────────────────────────────────
print("Running per-well regressions...")
reg_rows = []
for (wid, gid, gname), grp in data.groupby(["well_id","gage_id","gage_name"]):
    x, y = grp["z_wte"].values, grp["z_q"].values
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
reg_df.to_csv(FEAT_DIR / "regression_std_anom_by_well.csv", index=False)
print(f"  Saved regression ({len(reg_df)} rows)")

print("\n=== Standardized anomaly regression — summary ===")
print(f"{'Gage':<22} {'Wells':>6} {'%Neg':>6} {'%Sig':>6} "
      f"{'Median slope':>13} {'Median R²':>10} {'Median N':>9}")
print("-"*78)
for gname, grp in reg_df.groupby("gage_name"):
    short = GAGE_SHORT.get(gname,gname)[:20]
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
        sub = gdata[gdata["quarter"]==q].dropna(subset=["z_wte","z_q"])
        if not sub.empty:
            ax.scatter(sub["z_wte"], sub["z_q"],
                       s=10, color=color, alpha=0.35,
                       edgecolors="none", zorder=3, label=q)
    # slope=1 参考线（理想的同步响应）
    xlim_ref = (-3.5, 3.5)
    ax.plot(xlim_ref, xlim_ref, color="#999999", linewidth=1.2,
            linestyle=":", zorder=2, label="slope=1 (ref)")
    st = pooled_stats(gdata, "z_wte", "z_q")
    if st:
        xv = gdata["z_wte"].dropna().values
        x_fit = np.linspace(max(xv.min(),-4), min(xv.max(),4), 200)
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
    ax.set_xlim(-4,4); ax.set_ylim(-4,4)
    ax.set_xlabel("WTE standardized anomaly (σ)", fontsize=11)
    ax.set_ylabel("Q standardized anomaly (σ)", fontsize=11)
    ax.set_title(f"{gname}\nQ vs WTE — Standardized anomaly regression (detrended)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9,loc="upper left",framealpha=0.85)
    ax.grid(True,alpha=0.2)
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
    sub = data[(data["gage_name"]==gname)&(data["well_id"].isin(top_ids))
               ].dropna(subset=["z_wte","z_q"])
    present = [w for w in top_ids if w in set(sub["well_id"].unique())]
    if not present: continue
    n = len(present)
    ncols = min(5,n); nrows = math.ceil(n/ncols)
    fig, axes = plt.subplots(nrows,ncols,figsize=(4.2*ncols,4.0*nrows),squeeze=False)
    axes_flat = axes.flatten()
    for i, wid in enumerate(present):
        ax = axes_flat[i]
        wd = sub[sub["well_id"]==wid]
        for q, color in QUARTER_COLOR.items():
            qd = wd[wd["quarter"]==q]
            if not qd.empty:
                ax.scatter(qd["z_wte"],qd["z_q"],s=14,color=color,
                           alpha=0.7,edgecolors="none",zorder=3,label=q)
        x, y = wd["z_wte"].values, wd["z_q"].values
        stats = [f"n={len(x)}"]
        if len(x)>=MIN_FIT and x.std()>0:
            s,ic,r_val,p_val,_ = linregress(x,y)
            x_fit = np.linspace(-3.5,3.5,200)
            ax.plot(x_fit,s*x_fit+ic,color="black",linewidth=1.4,linestyle="--",zorder=4)
            p_str = f"{p_val:.3e}" if p_val<0.001 else f"{p_val:.3f}"
            stats += [f"slope={s:.3f}",f"R²={r_val**2:.3f}",f"p={p_str}"]
        ax.text(0.97,0.97,"\n".join(stats),transform=ax.transAxes,fontsize=7.5,
                ha="right",va="top",
                bbox=dict(boxstyle="round,pad=0.25",facecolor="white",alpha=0.85,edgecolor="#AAAAAA"))
        ax.axhline(0,color="#CCCCCC",linewidth=0.6,linestyle=":")
        ax.axvline(0,color="#CCCCCC",linewidth=0.6,linestyle=":")
        ax.set_xlim(-4,4); ax.set_ylim(-4,4)
        ax.set_xlabel("WTE anomaly (σ)",fontsize=8)
        ax.set_ylabel("Q anomaly (σ)",fontsize=8)
        mv = mvals.get(str(wid))
        ax.set_title(f"#{i+1}  {wid}"+(f"  R²={mv:.3f}" if mv else ""),fontsize=7.5,fontweight="bold")
        ax.grid(True,alpha=0.2)
        if i==0: ax.legend(fontsize=6.5,loc="upper left",framealpha=0.8)
    for j in range(n,len(axes_flat)): axes_flat[j].set_visible(False)
    fig.suptitle(f"{gname}\nQ vs WTE standardized anomaly per well (top {n} by R²)",
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
    sub = data[(data["gage_name"]==gname)&(data["well_id"].isin(top_ids))
               ].dropna(subset=["z_wte","z_q"])
    wells = sub["well_id"].unique()
    if not len(wells): continue
    fig,ax = plt.subplots(figsize=(9,6))
    all_x = sub["z_wte"].values; all_y = sub["z_q"].values
    x_fit = np.linspace(-3.5,3.5,200)
    legend_entries = []
    for i, wid in enumerate(wells):
        color = CMAP(i%10)
        wd = sub[sub["well_id"]==wid]
        x, y = wd["z_wte"].values, wd["z_q"].values
        sc = ax.scatter(x,y,s=12,color=color,alpha=0.55,edgecolors="none",zorder=3)
        label = str(wid)
        if len(x)>=MIN_FIT and x.std()>0:
            s,ic,r_val,p_val,_ = linregress(x,y)
            ax.plot(x_fit,s*x_fit+ic,color=color,linewidth=1.3,alpha=0.9,zorder=4)
            label += f"  (s={s:.2f}, R²={r_val**2:.3f})"
        legend_entries.append((sc,label))
    s_all,ic_all,r_all,p_all,_ = linregress(all_x,all_y)
    ax.plot(x_fit,s_all*x_fit+ic_all,color="black",linewidth=2,linestyle="--",zorder=6)
    # slope=1 参考线
    ax.plot([-3.5,3.5],[-3.5,3.5],color="#999999",linewidth=1,linestyle=":",zorder=2)
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
    ax.set_xlim(-4,4); ax.set_ylim(-4,4)
    ax.set_xlabel("WTE standardized anomaly (σ)",fontsize=11)
    ax.set_ylabel("Q standardized anomaly (σ)",fontsize=11)
    ax.set_title(f"{gname}\nStandardized anomaly regression (top {len(wells)}, per-well fit)",
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
pd.concat(tables).reset_index(drop=True).to_csv(TAB_DIR/"top10_std_anom_by_r2.csv",index=False)
print(f"\nAll outputs → {OUT_DIR}")
