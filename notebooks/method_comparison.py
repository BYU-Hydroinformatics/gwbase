"""
Method comparison summary: aggregate regression results from all delta methods
into one folder and generate comparison figures and a summary report.

Output: result/analysis/method_comparison/
  summary_table.csv            — pooled regression statistics for all methods × gages
  figures/
    comparison_pooled_slope.png  — pooled regression slope by method and gage
    comparison_r_squared.png     — pooled R² by method and gage
    comparison_pvalue.png        — pooled -log10(p_value) by method and gage
    comparison_heatmap.png       — global heatmap (method × gage × metric)
    slope_distributions/         — per-well slope distribution histogram per method
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "result"
OUT_DIR = RESULTS / "analysis" / "method_comparison"
FIG_DIR = OUT_DIR / "figures"
DIST_DIR = FIG_DIR / "slope_distributions"
for d in [OUT_DIR, FIG_DIR, DIST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GAGE_SHORT = {
    "BEAR RIVER NEAR CORINNE - UT":                  "Bear River",
    "WEBER RIVER NEAR PLAIN CITY - UT":               "Weber River",
    "PROVO RIVER AT PROVO - UT":                      "Provo River",
    "SPANISH FORK NEAR LAKE SHORE - UTAH":            "Spanish Fork",
    "LITTLE COTTONWOOD CREEK @ JORDAN RIVER NR SLC":  "Little Cottonwood",
}
GAGE_ORDER = list(GAGE_SHORT.values())

# ── Method definitions: name / delta data path / x column / y column ─────────
# Each method uses a pooled scatter regression (all wells under a gage combined
# into one OLS fit), corresponding to the dashed black line in scatter_by_gage
# plots — not a summary of per-well regression statistics.
METHODS = [
    ("Monthly delta\n(original)",
     RESULTS / "features" / "data_with_deltas.csv",
     "delta_wte", "delta_q"),
    ("Annual delta\n(YoY)",
     RESULTS / "delta_methods" / "annual" / "features" / "data_annual_deltas.csv",
     "delta_wte_ann", "delta_q_ann"),
    ("Same-quarter\nYoY delta",
     RESULTS / "delta_methods" / "quarterly_yoy" / "features" / "data_quarterly_deltas.csv",
     "delta_wte_qtr", "delta_q_qtr"),
    ("Deseason qtr\nconsec delta",
     RESULTS / "delta_methods" / "deseason_qtr" / "features" / "data_deseason_qtr_deltas.csv",
     "delta_wte", "delta_q"),
    ("Rolling 12m\nannual diff",
     RESULTS / "delta_methods" / "rolling12m" / "features" / "data_rolling12m_deltas.csv",
     "delta_wte", "delta_q"),
    ("Std anomaly\nregression",
     RESULTS / "delta_methods" / "std_anomaly" / "features" / "data_std_anom.csv",
     "z_wte", "z_q"),
]

METHOD_COLORS = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))

# ── Load delta data and compute pooled regression per method × gage ──────────
from scipy.stats import linregress

print("Loading delta data and computing pooled regressions...")
all_dfs   = {}   # label → delta DataFrame (used for per-well slope distribution plots)
summary_rows = []

for (label, path, xcol, ycol) in METHODS:
    if not path.exists():
        print(f"  MISSING: {path}")
        continue
    df = pd.read_csv(path)
    df["gage_short"] = df["gage_name"].map(GAGE_SHORT)

    # Compute per-well regression directly from delta data for slope distribution plots
    reg_path = path.parent / path.name.replace("data_", "regression_").replace(
        "_deltas.csv", "_by_well.csv").replace("data_with_deltas.csv",
        "../features/regression_by_well.csv")
    # Simple approach: compute per-well linregress directly from the delta data
    well_slopes = {}
    for (wid, gid, gname), grp in df.groupby(["well_id","gage_id","gage_name"]):
        sub = grp[[xcol,ycol]].dropna()
        x, y = sub[xcol].values, sub[ycol].values
        if len(x) >= 5 and x.std() > 0:
            s, _, _, _, _ = linregress(x, y)
            well_slopes[(wid, gid, gname)] = s
    well_df = pd.DataFrame([
        {"well_id":k[0],"gage_id":k[1],"gage_name":k[2],"slope":v,
         "gage_short": GAGE_SHORT.get(k[2])}
        for k, v in well_slopes.items()
    ])
    all_dfs[label] = well_df

    # ── Pooled regression per gage ──────────────────────────────────────────
    for gshort in GAGE_ORDER:
        sub = df[df["gage_short"]==gshort][[xcol,ycol]].dropna()
        x, y = sub[xcol].values, sub[ycol].values
        if len(x) < 5 or x.std() == 0:
            continue
        s, ic, r_val, p_val, _ = linregress(x, y)
        summary_rows.append({
            "method":    label.replace("\n"," "),
            "gage":      gshort,
            "n_pts":     len(x),
            "n_wells":   df[df["gage_short"]==gshort]["well_id"].nunique(),
            "slope":     round(s, 4),
            "intercept": round(ic, 4),
            "r_squared": round(r_val**2, 4),
            "p_value":   round(p_val, 6),
            "slope_pos": s > 0,
        })

    n_gages = len([g for g in GAGE_ORDER if not df[df["gage_short"]==g].empty])
    print(f"  {label.replace(chr(10),' '):<35}  {df['well_id'].nunique():>5} wells, "
          f"{n_gages} gages")

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "summary_table.csv", index=False)
print(f"  Saved summary_table.csv  ({len(summary)} rows)")

# ── Generic grouped bar chart function (based on pooled regression metrics) ───
def grouped_bar(metric, ylabel, title, outname, ylim=None, hline=None):
    method_labels = list(all_dfs.keys())
    n_methods = len(method_labels)
    n_gages   = len(GAGE_ORDER)
    bar_w = 0.72 / n_methods
    x = np.arange(n_gages)

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, label in enumerate(method_labels):
        vals = []
        for g in GAGE_ORDER:
            row = summary[(summary["method"]==label.replace("\n"," ")) &
                          (summary["gage"]==g)]
            vals.append(row[metric].values[0] if len(row) else np.nan)
        offset = (i - n_methods/2 + 0.5) * bar_w
        # color bars red if slope is negative (for slope metric)
        if metric == "slope":
            bar_colors = [
                "#E24A33" if (v is not None and not np.isnan(v) and v < 0) else METHOD_COLORS[i]
                for v in vals
            ]
            for j, (v, c) in enumerate(zip(vals, bar_colors)):
                ax.bar(x[j] + offset, v, width=bar_w*0.92,
                       color=c, alpha=0.85, zorder=3,
                       label=label.replace("\n"," ") if j==0 else "")
        else:
            ax.bar(x + offset, vals, width=bar_w*0.92,
                   color=METHOD_COLORS[i], alpha=0.85,
                   label=label.replace("\n"," "), zorder=3)

    if hline is not None:
        ax.axhline(hline, color="red", linewidth=1.2, linestyle="--",
                   alpha=0.6, label=f"y={hline}", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(GAGE_ORDER, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if ylim: ax.set_ylim(ylim)
    ax.legend(fontsize=7.5, ncol=4, loc="upper right",
              framealpha=0.9, bbox_to_anchor=(1, 1))
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = FIG_DIR / outname
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out}")

print("\nGenerating comparison bar charts (pooled regression)...")
grouped_bar("slope",     "Pooled regression slope (ΔQ / ΔWTE)",
            "Pooled scatter slope by method and gage\n(positive = gaining stream; red bars = negative slopes)",
            "comparison_pooled_slope.png", hline=0)

grouped_bar("r_squared", "Pooled R²",
            "Pooled R² by method and gage\n(higher = tighter WTE–Q relationship)",
            "comparison_r_squared.png", ylim=(0, None))

# Compute -log10(p_value) for the p-value bar chart
summary["neg_log10_p"] = -np.log10(summary["p_value"].clip(lower=1e-300))
grouped_bar("neg_log10_p", "−log₁₀(p-value)  [higher = more significant]",
            "Pooled regression significance by method and gage\n(dashed = p=0.05 threshold)",
            "comparison_pvalue.png", hline=-np.log10(0.05))

# ── Heatmap: method × gage, three metrics as separate columns ────────────────
print("Generating heatmap...")
method_labels = list(all_dfs.keys())
short_labels  = [m.replace("\n"," ") for m in method_labels]

# Symmetric colormap limits for slope
slope_abs_max = summary["slope"].abs().quantile(0.95) if len(summary) else 1.0
r2_max = summary["r_squared"].quantile(0.95) if len(summary) else 1.0

metrics = [
    ("slope",        "Pooled slope",      plt.cm.RdYlGn,   -slope_abs_max, slope_abs_max),
    ("r_squared",    "Pooled R²",         plt.cm.YlOrRd,    0,             r2_max),
    ("neg_log10_p",  "−log₁₀(p)",         plt.cm.YlOrRd,    0,             None),
]

fig = plt.figure(figsize=(20, 8))
fig.suptitle("Method comparison heatmap across all gages\n(pooled scatter regression per method × gage)",
             fontsize=13, fontweight="bold", y=1.01)

gs = gridspec.GridSpec(1, 3, wspace=0.35)

for col, (metric, mtitle, cmap, vmin, vmax) in enumerate(metrics):
    ax = fig.add_subplot(gs[0, col])
    mat = np.full((len(short_labels), len(GAGE_ORDER)), np.nan)
    for i, lbl in enumerate(short_labels):
        for j, g in enumerate(GAGE_ORDER):
            row = summary[(summary["method"]==lbl) & (summary["gage"]==g)]
            if len(row):
                mat[i, j] = row[metric].values[0]

    vmax_use = vmax if vmax is not None else (np.nanmax(mat) * 1.05 if not np.all(np.isnan(mat)) else 1)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax_use, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    ax.set_xticks(range(len(GAGE_ORDER)))
    ax.set_xticklabels([g.split()[0] for g in GAGE_ORDER], fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_title(mtitle, fontsize=11, fontweight="bold")

    # Annotate each cell with its numeric value
    for i in range(len(short_labels)):
        for j in range(len(GAGE_ORDER)):
            v = mat[i, j]
            if not np.isnan(v):
                if metric == "slope":
                    txt = f"{v:.3f}"
                elif metric == "r_squared":
                    txt = f"{v:.3f}"
                else:
                    txt = f"{v:.1f}"
                mid = (vmin + vmax_use) / 2 if vmin is not None else vmax_use * 0.5
                fc = "white" if v < mid else "black"
                if metric == "slope":
                    # diverging: white near 0, dark near extremes
                    fc = "black" if abs(v) < slope_abs_max * 0.4 else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=fc)

plt.tight_layout()
out = FIG_DIR / "comparison_heatmap.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved → {out}")

# ── Slope distributions: one row per method, one column per gage ─────────────
print("Generating slope distribution figures per method...")
for label, df in all_dfs.items():
    fname = label.replace("\n","_").replace(" ","_").replace("/","").lower() + ".png"
    fig, axes = plt.subplots(1, len(GAGE_ORDER), figsize=(18, 3.8),
                              gridspec_kw={"wspace":0.4})
    fig.suptitle(f"Slope distribution — {label.replace(chr(10),' ')}",
                 fontsize=11, fontweight="bold")
    for ax, gshort in zip(axes, GAGE_ORDER):
        grp = df[df["gage_short"]==gshort]["slope"].dropna()
        if len(grp) < 3:
            ax.set_title(f"{gshort}\n(n<3)", fontsize=8)
            ax.set_visible(True)
            continue
        p2, p98 = np.percentile(grp, 2), np.percentile(grp, 98)
        sc = np.clip(grp, p2, p98)
        bins = min(30, max(6, len(sc)//3))
        counts, edges, patches = ax.hist(sc, bins=bins,
                                          edgecolor="white", linewidth=0.3)
        for patch, left in zip(patches, edges[:-1]):
            patch.set_facecolor("#E24A33" if left < 0 else "#4C72B0")
            patch.set_alpha(0.78)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        n_neg = (grp < 0).sum()
        n_pos = (grp > 0).sum()
        ax.set_title(f"{gshort}\nn={len(grp)}  neg={n_neg}({100*n_neg/len(grp):.0f}%)",
                     fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Slope", fontsize=8)
    plt.tight_layout()
    out = DIST_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
print(f"  Saved {len(all_dfs)} slope distribution figures → {DIST_DIR}")

# ── Per-gage method comparison figure (one column per gage) ──────────────────
print("Generating per-gage method comparison figure...")

fig, axes = plt.subplots(1, len(GAGE_ORDER), figsize=(22, 6),
                          gridspec_kw={"wspace":0.45})
fig.suptitle("Pooled regression: slope / R² / significance by method and gage",
             fontsize=11, fontweight="bold")

for ax, gshort in zip(axes, GAGE_ORDER):
    sub = summary[summary["gage"]==gshort].copy()
    # maintain consistent method order
    order = [m.replace("\n"," ") for m in all_dfs.keys()]
    sub["_ord"] = sub["method"].map({m: i for i, m in enumerate(order)})
    sub = sub.sort_values("_ord").reset_index(drop=True)

    n = len(sub)
    if n == 0:
        ax.set_visible(False)
        continue

    y = np.arange(n)
    colors = ["#4C72B0" if s > 0 else "#E24A33" for s in sub["slope"]]
    bars = ax.barh(y, sub["slope"], color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")

    # annotate R² and p-value
    for yi, (_, row) in enumerate(sub.iterrows()):
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01
              else ("*" if row["p_value"] < 0.05 else ""))
        note = f"R²={row['r_squared']:.3f}{sig}"
        x_annot = row["slope"]
        ha = "left" if x_annot >= 0 else "right"
        pad = ax.get_xlim()[1] * 0.02 if ax.get_xlim()[1] != 0 else 0.001
        ax.text(x_annot + (pad if x_annot >= 0 else -pad), yi,
                note, va="center", ha=ha, fontsize=5.5)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["method"], fontsize=6.5)
    ax.set_title(gshort, fontsize=9, fontweight="bold")
    ax.set_xlabel("Pooled slope", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(axis="x", labelsize=7)

plt.tight_layout()
out = FIG_DIR / "comparison_scores_per_gage.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved → {out}")

print(f"\nAll outputs → {OUT_DIR}")
print(f"\n{'='*70}")
print("SUMMARY TABLE — pooled regression (mean across gages)")
print(f"{'='*70}")
agg = (summary.groupby("method")
       .agg(avg_slope  =("slope",     "mean"),
            pct_pos    =("slope_pos", lambda x: 100*x.mean()),
            avg_r2     =("r_squared", "mean"),
            avg_neglogp=("neg_log10_p","mean"),
            total_pts  =("n_pts",     "sum"))
       .sort_values("avg_slope", ascending=False))
print(f"\n{'Method':<37} {'AvgSlope':>10} {'%PosGages':>10} {'AvgR²':>8} {'-log10p':>9} {'Pts':>8}")
print("-"*85)
for idx, row in agg.iterrows():
    print(f"{idx:<37} {row['avg_slope']:>10.4f} {row['pct_pos']:>9.0f}% "
          f"{row['avg_r2']:>8.4f} {row['avg_neglogp']:>9.1f} {row['total_pts']:>8.0f}")
