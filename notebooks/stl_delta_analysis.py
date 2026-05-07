"""
STL去季节化差分回归（Seasonal-Trend decomposition using Loess）

文献背景：
  STL是Cleveland et al. (1990)提出的经典时间序列分解方法，近年大量水文研究用于
  地下水和径流的去季节化（如HESS 2022, Journal of Hydrology 2020+）。
  优于简单减均值：季节形态可随时间变化（loess窗口局部拟合），对非平稳序列更鲁棒。

方法：
  1. 对每个 well-gage 对的 WTE 和 Q 月度序列各自做 STL 分解
     → seasonal（季节）+ trend（趋势）+ remainder（残差）
  2. 去季节化序列 = trend + remainder（即原始值 - seasonal）
  3. 对去季节化序列做同季度年际差分（Q1_t - Q1_{t-1}），消除残余趋势影响
  4. 回归 Δ(deseason Q) ~ Δ(deseason WTE)

  注：STL需要连续时间序列，对缺口较多的 well-gage 对进行线性插值（≤6个月缺口）。

结果目录：results/stl_delta/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import warnings
from scipy.stats import linregress, pearsonr, spearmanr
from statsmodels.tsa.seasonal import STL
from pathlib import Path

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "stl_delta"

FEAT_DIR = OUT_DIR / "features"
FIG_DIR  = OUT_DIR / "figures"
SCAT_DIR = FIG_DIR / "scatter_by_gage"
T10_DIR  = FIG_DIR / "top10_by_r2"
T10F_DIR = FIG_DIR / "top10_by_r2_fit"
TAB_DIR  = FIG_DIR / "tables"
for d in [FEAT_DIR, FIG_DIR, SCAT_DIR, T10_DIR, T10F_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MIN_OBS   = 8
MIN_YEARS = 3
MIN_FIT   = 5
TOP_N     = 10
MAX_GAP   = 6    # 最大允许插值缺口（月数）
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

# ── STL decomposition per well-gage pair ──────────────────────────────────────
print("Applying STL decomposition per well-gage pair...")

def stl_deseason(series_monthly):
    """
    对月度序列做 STL 分解，返回去季节化序列（trend + remainder）。
    输入：以 date 为索引的 Series（允许有缺失值）。
    """
    # 建立完整月度索引
    full_idx = pd.date_range(series_monthly.index.min(),
                             series_monthly.index.max(), freq="MS")
    s = series_monthly.reindex(full_idx)

    # 统计缺口
    gap_runs = (s.isna().astype(int)
                .groupby((s.notna().astype(int).cumsum()))
                .sum())
    max_gap = gap_runs.max() if len(gap_runs) else 0
    if max_gap > MAX_GAP:
        return None   # 缺口太大，跳过

    # 线性插值填补短缺口
    s = s.interpolate(method="linear", limit=MAX_GAP)
    if s.isna().any():
        s = s.ffill().bfill()

    if len(s) < 24:
        return None

    try:
        result = STL(s, period=12, seasonal=13, robust=True).fit()
        deseason = result.trend + result.resid
        return pd.Series(deseason, index=full_idx)
    except Exception:
        return None

rows_out = []
n_pairs = data.groupby(["well_id","gage_id"]).ngroups
n_done = 0
n_skip = 0

for (wid, gid, gname), grp in data.groupby(["well_id","gage_id","gage_name"]):
    grp = grp.sort_values("date").copy()
    grp["period"] = grp["date"].dt.to_period("M").dt.to_timestamp()  # month-start
    grp = grp.set_index("period")

    ds_wte = stl_deseason(grp["wte"])
    ds_q   = stl_deseason(grp["q"])
    if ds_wte is None or ds_q is None:
        n_skip += 1
        continue

    combined = pd.DataFrame({"wte_ds": ds_wte, "q_ds": ds_q}).dropna()
    combined["year"]    = combined.index.year
    combined["month"]   = combined.index.month
    combined["quarter"] = combined["month"].map(QUARTER_MAP)

    for row in combined.itertuples():
        rows_out.append({"well_id":wid,"gage_id":gid,"gage_name":gname,
                         "date":row.Index,"year":row.year,
                         "month":row.month,"quarter":row.quarter,
                         "wte_ds":row.wte_ds,"q_ds":row.q_ds})
    n_done += 1

ds_df = pd.DataFrame(rows_out)
print(f"  STL decomposed: {n_done} pairs, skipped: {n_skip}")

# ── Same-quarter year-over-year delta on deseasonalized values ────────────────
ds_df = ds_df.sort_values(["well_id","gage_id","quarter","year"]).reset_index(drop=True)
ds_df["delta_wte"] = ds_df.groupby(["well_id","gage_id","quarter"])["wte_ds"].diff()
ds_df["delta_q"]   = ds_df.groupby(["well_id","gage_id","quarter"])["q_ds"].diff()
ds_df["year_diff"] = ds_df.groupby(["well_id","gage_id","quarter"])["year"].diff()

ds_df = ds_df[ds_df["year_diff"]==1].dropna(subset=["delta_wte","delta_q"])

# Outlier removal
def iqr_mask(series, k=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0: return pd.Series(True, index=series.index)
    return (series >= q1-k*iqr) & (series <= q3+k*iqr)

before = len(ds_df)
ds_df = ds_df[ds_df.groupby(["well_id","gage_id"])["delta_wte"]
              .transform(iqr_mask)].copy()
print(f"  Outlier removal: {before-len(ds_df)} rows removed")
ds_df.to_csv(FEAT_DIR / "data_stl_deltas.csv", index=False)
print(f"  Saved ({len(ds_df)} rows)")

# ── Per-well regression ───────────────────────────────────────────────────────
print("Running per-well regressions...")
reg_rows = []
for (wid, gid, gname), grp in ds_df.groupby(["well_id","gage_id","gage_name"]):
    x, y = grp["delta_wte"].values, grp["delta_q"].values
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
reg_df.to_csv(FEAT_DIR / "regression_stl_by_well.csv", index=False)
print(f"  Saved regression ({len(reg_df)} rows)")

print("\n=== STL deseasonalized same-quarter Δ — regression summary ===")
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
    gdata = ds_df[ds_df["gage_name"]==gname]
    if gdata.empty: continue
    fig, ax = plt.subplots(figsize=(9,6))
    for q, color in QUARTER_COLOR.items():
        sub = gdata[gdata["quarter"]==q].dropna(subset=["delta_wte","delta_q"])
        if not sub.empty:
            ax.scatter(sub["delta_wte"], sub["delta_q"],
                       s=18, color=color, alpha=0.5,
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
    ax.set_xlabel("Δ(STL-deseason WTE) ft/yr", fontsize=11)
    ax.set_ylabel("Δ(STL-deseason Q) cfs/yr", fontsize=11)
    ax.set_title(f"{gname}\nΔQ vs ΔWTE — STL deseasonalized, same-quarter YoY diff",
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
    sub = ds_df[(ds_df["gage_name"]==gname)&(ds_df["well_id"].isin(top_ids))
                ].dropna(subset=["delta_wte","delta_q"])
    present = [w for w in top_ids if w in set(sub["well_id"].unique())]
    if not present: continue
    n = len(present)
    ncols = min(5,n); nrows = math.ceil(n/ncols)
    fig, axes = plt.subplots(nrows,ncols,figsize=(4.2*ncols,4.0*nrows),squeeze=False)
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
                ax.scatter(qd["delta_wte"],qd["delta_q"],s=22,color=color,
                           alpha=0.8,edgecolors="none",zorder=3,label=q)
        x, y = wd["delta_wte"].values, wd["delta_q"].values
        stats = [f"n={len(x)}"]
        if len(x)>=MIN_FIT and x.std()>0:
            s,ic,r_val,p_val,_ = linregress(x,y)
            x_fit = np.linspace(xlim[0],xlim[1],200)
            ax.plot(x_fit,s*x_fit+ic,color="black",linewidth=1.4,linestyle="--",zorder=4)
            p_str = f"{p_val:.3e}" if p_val<0.001 else f"{p_val:.3f}"
            stats += [f"slope={s:.3f}",f"R²={r_val**2:.3f}",f"p={p_str}"]
        ax.text(0.97,0.97,"\n".join(stats),transform=ax.transAxes,fontsize=7.5,
                ha="right",va="top",
                bbox=dict(boxstyle="round,pad=0.25",facecolor="white",alpha=0.85,edgecolor="#AAAAAA"))
        ax.axhline(0,color="#CCCCCC",linewidth=0.6,linestyle=":")
        ax.axvline(0,color="#CCCCCC",linewidth=0.6,linestyle=":")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("ΔWTE STL (ft)",fontsize=8)
        ax.set_ylabel("ΔQ STL (cfs)",fontsize=8)
        mv = mvals.get(str(wid))
        ax.set_title(f"#{i+1}  {wid}"+(f"  R²={mv:.3f}" if mv else ""),fontsize=7.5,fontweight="bold")
        ax.grid(True,alpha=0.2)
        if i==0: ax.legend(fontsize=6.5,loc="upper left",framealpha=0.8)
    for j in range(n,len(axes_flat)): axes_flat[j].set_visible(False)
    fig.suptitle(f"{gname}\nΔQ vs ΔWTE per well (top {n} by R², STL deseasonalized)",
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
    sub = ds_df[(ds_df["gage_name"]==gname)&(ds_df["well_id"].isin(top_ids))
                ].dropna(subset=["delta_wte","delta_q"])
    wells = sub["well_id"].unique()
    if not len(wells): continue
    fig,ax = plt.subplots(figsize=(9,6))
    all_x = sub["delta_wte"].values; all_y = sub["delta_q"].values
    xpad = (all_x.max()-all_x.min())*0.05 or 1
    xlim = (all_x.min()-xpad, all_x.max()+xpad)
    x_fit = np.linspace(xlim[0],xlim[1],200)
    legend_entries = []
    for i, wid in enumerate(wells):
        color = CMAP(i%10)
        wd = sub[sub["well_id"]==wid]
        x, y = wd["delta_wte"].values, wd["delta_q"].values
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
    ax.set_xlabel("Δ(STL-deseason WTE) ft/yr",fontsize=11)
    ax.set_ylabel("Δ(STL-deseason Q) cfs/yr",fontsize=11)
    ax.set_title(f"{gname}\nSTL deseasonalized Δ (top {len(wells)}, per-well fit)",
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
pd.concat(tables).reset_index(drop=True).to_csv(TAB_DIR/"top10_stl_by_r2.csv",index=False)
print(f"\nAll outputs → {OUT_DIR}")
