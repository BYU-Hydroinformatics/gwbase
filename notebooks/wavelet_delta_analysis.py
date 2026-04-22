"""
小波分解年际分量回归（Discrete Wavelet Transform interannual component）

文献背景：
  离散小波变换（DWT）是水文时间序列多尺度分析的经典方法（Labat et al. 2004,
  Adamowski & Sun 2010, Sang 2013 J. Hydrol）。与STL不同，小波把信号分解到
  不同"频率层"（level），每个 level 对应不同时间尺度：
    月度数据（dt=1月），Daubechies-4小波（db4）：
      Level 1 detail  → 2–4 个月的变化
      Level 2 detail  → 4–8 个月的变化（季节内）
      Level 3 detail  → 8–16 个月的变化（跨季节/年度）
      Level 4 detail  → 16–32 个月（1–3年）← 年际信号
      Level 5 detail  → 32–64 个月（3–5年）← 多年际
      Approximation   → >64 个月（长期趋势）

方法：
  1. 对每个 well-gage 对的 WTE 和 Q 月度序列做 N 层 DWT（db4 小波）
  2. 重建"年际分量"= Level 4 + Level 5 + Approximation（去掉季节和月度噪声）
     即：去掉 Detail level 1、2、3（< 16 个月的高频变化）
  3. 对重建的年际序列做同季度年际差分（消除残余趋势）
  4. 回归 Δ(interannual Q) ~ Δ(interannual WTE)

注：
  - DWT 要求序列长度为 2^N 的倍数；用零填充（zero-padding）至最近的 2^N，
    变换后截回原始长度
  - 最少需要 4 个 Level（要求序列长度 ≥ 2^4 = 16 个月）
  - 使用 db4（Daubechies-4）：水文常用小波，平滑性好

结果目录：results/wavelet_delta/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import warnings
import pywt
from scipy.stats import linregress, pearsonr, spearmanr
from pathlib import Path

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "wavelet_delta"

FEAT_DIR = OUT_DIR / "features"
FIG_DIR  = OUT_DIR / "figures"
SCAT_DIR = FIG_DIR / "scatter_by_gage"
T10_DIR  = FIG_DIR / "top10_by_r2"
T10F_DIR = FIG_DIR / "top10_by_r2_fit"
TAB_DIR  = FIG_DIR / "tables"
for d in [FEAT_DIR, FIG_DIR, SCAT_DIR, T10_DIR, T10F_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WAVELET    = "db4"
N_LEVELS   = 5          # 总分解层数
KEEP_FROM  = 4          # 保留 Level >= KEEP_FROM 的 detail + approximation
                        # → 去掉 Level 1–3（< 16 个月的高频变化）
MAX_GAP    = 6          # 最大允许插值缺口（月）
MIN_LENGTH = 2**N_LEVELS * 2   # 最短序列（月），确保有意义的分解

MIN_OBS   = 8
MIN_YEARS = 3
MIN_FIT   = 5
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

# ── 小波重建函数 ──────────────────────────────────────────────────────────────
def wavelet_interannual(series_monthly, wavelet=WAVELET, n_levels=N_LEVELS,
                        keep_from=KEEP_FROM, max_gap=MAX_GAP):
    """
    对月度序列做 N 层 DWT，重建去掉高频分量（level 1 到 keep_from-1）后的年际信号。
    返回与原始序列等长的 Series（index 与输入一致），或 None（序列太短/缺口过大）。
    """
    # 建立完整月度索引（月初）
    full_idx = pd.date_range(series_monthly.index.min(),
                             series_monthly.index.max(), freq="MS")
    s = series_monthly.reindex(full_idx)

    # 检查最大缺口
    is_na = s.isna().values
    gap_len, max_gap_found = 0, 0
    for v in is_na:
        if v: gap_len += 1
        else: max_gap_found = max(max_gap_found, gap_len); gap_len = 0
    max_gap_found = max(max_gap_found, gap_len)
    if max_gap_found > max_gap:
        return None

    # 线性插值填补短缺口
    s = s.interpolate(method="linear", limit=max_gap)
    if s.isna().any():
        s = s.ffill().bfill()

    n = len(s)
    if n < MIN_LENGTH:
        return None

    # 零填充至最近的 2^ceil(log2(n))
    n_pad = int(2 ** np.ceil(np.log2(n)))
    signal = np.zeros(n_pad)
    signal[:n] = s.values

    # DWT 分解
    coeffs = pywt.wavedec(signal, wavelet, level=n_levels)
    # coeffs[0]   = approximation（最低频）
    # coeffs[1]   = detail level N_LEVELS（最低频 detail）
    # coeffs[-1]  = detail level 1（最高频）

    # 置零高频分量（level 1 到 keep_from-1）
    # coeffs 索引：[approx, detailN, detailN-1, ..., detail1]
    # detail level k 在 coeffs[N_LEVELS - k + 1]
    for k in range(1, keep_from):          # zero out level 1 .. keep_from-1
        idx = n_levels - k + 1
        if idx < len(coeffs):
            coeffs[idx] = np.zeros_like(coeffs[idx])

    # 重建
    reconstructed = pywt.waverec(coeffs, wavelet)[:n]
    return pd.Series(reconstructed, index=full_idx)

# ── 对每个 well-gage 对应用小波分解 ──────────────────────────────────────────
print(f"Applying {WAVELET} wavelet decomposition (level {N_LEVELS}, "
      f"keep level {KEEP_FROM}+) per well-gage pair...")

rows_out = []
n_done, n_skip = 0, 0

for (wid, gid, gname), grp in data.groupby(["well_id","gage_id","gage_name"]):
    grp = grp.sort_values("date").copy()
    grp["period"] = grp["date"].dt.to_period("M").dt.to_timestamp()
    grp = grp.set_index("period")

    wt_wte = wavelet_interannual(grp["wte"])
    wt_q   = wavelet_interannual(grp["q"])
    if wt_wte is None or wt_q is None:
        n_skip += 1
        continue

    combined = pd.DataFrame({"wte_wt": wt_wte, "q_wt": wt_q}).dropna()
    combined["year"]    = combined.index.year
    combined["month"]   = combined.index.month
    combined["quarter"] = combined["month"].map(QUARTER_MAP)

    for row in combined.itertuples():
        rows_out.append({"well_id":wid,"gage_id":gid,"gage_name":gname,
                         "date":row.Index,"year":row.year,
                         "month":row.month,"quarter":row.quarter,
                         "wte_wt":row.wte_wt,"q_wt":row.q_wt})
    n_done += 1

wt_df = pd.DataFrame(rows_out)
print(f"  Wavelet decomposed: {n_done} pairs, skipped: {n_skip}")

# ── 同季度年际差分（对年际重建序列） ─────────────────────────────────────────
wt_df = wt_df.sort_values(["well_id","gage_id","quarter","year"]).reset_index(drop=True)
wt_df["delta_wte"] = wt_df.groupby(["well_id","gage_id","quarter"])["wte_wt"].diff()
wt_df["delta_q"]   = wt_df.groupby(["well_id","gage_id","quarter"])["q_wt"].diff()
wt_df["year_diff"] = wt_df.groupby(["well_id","gage_id","quarter"])["year"].diff()

wt_df = wt_df[wt_df["year_diff"]==1].dropna(subset=["delta_wte","delta_q"])

# Outlier removal
def iqr_mask(series, k=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0: return pd.Series(True, index=series.index)
    return (series >= q1-k*iqr) & (series <= q3+k*iqr)

before = len(wt_df)
wt_df = wt_df[wt_df.groupby(["well_id","gage_id"])["delta_wte"]
              .transform(iqr_mask)].copy()
print(f"  Outlier removal: {before-len(wt_df)} rows removed")
wt_df.to_csv(FEAT_DIR / "data_wavelet_deltas.csv", index=False)
print(f"  Saved ({len(wt_df)} rows)")

# ── Per-well regression ───────────────────────────────────────────────────────
print("Running per-well regressions...")
reg_rows = []
for (wid, gid, gname), grp in wt_df.groupby(["well_id","gage_id","gage_name"]):
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
reg_df.to_csv(FEAT_DIR / "regression_wavelet_by_well.csv", index=False)
print(f"  Saved regression ({len(reg_df)} rows)")

print(f"\n=== Wavelet ({WAVELET}, level {N_LEVELS}, keep ≥{KEEP_FROM}) Δ — regression summary ===")
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
    gdata = wt_df[wt_df["gage_name"]==gname]
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
    ax.set_xlabel(f"Δ(wavelet interannual WTE) ft/yr", fontsize=11)
    ax.set_ylabel(f"Δ(wavelet interannual Q) cfs/yr", fontsize=11)
    ax.set_title(f"{gname}\nΔQ vs ΔWTE — {WAVELET} wavelet interannual component "
                 f"(level ≥{KEEP_FROM}, all wells)",
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
    sub = wt_df[(wt_df["gage_name"]==gname)&(wt_df["well_id"].isin(top_ids))
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
        ax.set_xlabel("ΔWTE wavelet (ft)",fontsize=8)
        ax.set_ylabel("ΔQ wavelet (cfs)",fontsize=8)
        mv = mvals.get(str(wid))
        ax.set_title(f"#{i+1}  {wid}"+(f"  R²={mv:.3f}" if mv else ""),fontsize=7.5,fontweight="bold")
        ax.grid(True,alpha=0.2)
        if i==0: ax.legend(fontsize=6.5,loc="upper left",framealpha=0.8)
    for j in range(n,len(axes_flat)): axes_flat[j].set_visible(False)
    fig.suptitle(f"{gname}\nΔQ vs ΔWTE per well (top {n} by R², {WAVELET} wavelet interannual)",
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
    sub = wt_df[(wt_df["gage_name"]==gname)&(wt_df["well_id"].isin(top_ids))
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
    ax.set_xlabel(f"Δ(wavelet interannual WTE) ft/yr",fontsize=11)
    ax.set_ylabel(f"Δ(wavelet interannual Q) cfs/yr",fontsize=11)
    ax.set_title(f"{gname}\n{WAVELET} wavelet interannual Δ (top {len(wells)}, per-well fit)",
                 fontsize=10,fontweight="bold")
    ax.grid(True,alpha=0.25)
    plt.tight_layout()
    out = T10F_DIR / f"{gname.split()[0].lower()}.png"
    plt.savefig(out,dpi=160,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

# ── 额外：对一个代表性 well-gage 对画小波分解示意图 ──────────────────────────
print("\nGenerating wavelet decomposition illustration...")
# 选数据最长的 Bear River well
bear_pairs = data[data["gage_name"]=="BEAR RIVER NEAR CORINNE - UT"]
if not bear_pairs.empty:
    longest = (bear_pairs.groupby(["well_id","gage_id"]).size()
               .sort_values(ascending=False).index[0])
    wid_ex, gid_ex = longest
    ex = bear_pairs[(bear_pairs["well_id"]==wid_ex)].sort_values("date").copy()
    ex["period"] = ex["date"].dt.to_period("M").dt.to_timestamp()
    ex = ex.set_index("period")

    full_idx = pd.date_range(ex.index.min(), ex.index.max(), freq="MS")
    s_raw = ex["wte"].reindex(full_idx).interpolate(limit=MAX_GAP).ffill().bfill()
    s_wt  = wavelet_interannual(ex["wte"])

    if s_wt is not None:
        fig, axes = plt.subplots(3, 1, figsize=(13, 9),
                                  gridspec_kw={"hspace":0.45})
        axes[0].plot(full_idx, s_raw, color="#4C72B0", linewidth=0.9, alpha=0.85)
        axes[0].set_title("Original monthly WTE", fontsize=10, fontweight="bold")
        axes[0].set_ylabel("WTE (ft)")
        axes[0].grid(True, alpha=0.2)

        axes[1].plot(s_wt.index, s_wt.values, color="#E24A33", linewidth=1.4)
        axes[1].set_title(f"Wavelet interannual component ({WAVELET}, level ≥{KEEP_FROM})",
                          fontsize=10, fontweight="bold")
        axes[1].set_ylabel("WTE interannual (ft)")
        axes[1].grid(True, alpha=0.2)

        high_freq = s_raw.values[:len(s_wt)] - s_wt.values
        axes[2].plot(s_wt.index, high_freq, color="#55A868", linewidth=0.8, alpha=0.7)
        axes[2].set_title(f"Removed high-frequency component (seasonal + noise, level <{KEEP_FROM})",
                          fontsize=10, fontweight="bold")
        axes[2].set_ylabel("WTE high-freq (ft)")
        axes[2].grid(True, alpha=0.2)
        axes[2].set_xlabel("Date")

        fig.suptitle(f"Wavelet decomposition illustration\nWell {wid_ex} — Bear River",
                     fontsize=11, fontweight="bold")
        out_ill = FIG_DIR / "wavelet_decomposition_example.png"
        plt.savefig(out_ill, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved → {out_ill}")

tables = []
for gname,grp in reg_df.groupby("gage_name"):
    ranked = grp.nlargest(TOP_N,"r_squared").copy()
    ranked.insert(0,"rank",range(1,len(ranked)+1))
    tables.append(ranked)
pd.concat(tables).reset_index(drop=True).to_csv(
    TAB_DIR/"top10_wavelet_by_r2.csv",index=False)
print(f"\nAll outputs → {OUT_DIR}")
