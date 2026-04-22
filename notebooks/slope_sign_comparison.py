"""
Visualise differences between negative-slope and positive-slope wells
(top-10 R² / MI tables).

Output:
  results/figures/slope_sign_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress, mannwhitneyu
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT     = RESULTS / "figures" / "slope_sign_comparison.png"

# ── Load data ────────────────────────────────────────────────────────────────
r2_df  = pd.read_csv(RESULTS / "figures" / "top10_wells_scatter" / "tables" / "top10_by_r2.csv")
data   = pd.read_csv(RESULTS / "features" / "data_with_deltas.csv", parse_dates=["date"])
pchip  = pd.read_csv(RESULTS / "processed" / "well_pchip_monthly.csv", parse_dates=["date"])

r2_df["well_id"] = r2_df["well_id"].astype(str)
r2_df["gage_id"] = r2_df["gage_id"].astype(str)
data["well_id"]  = data["well_id"].astype(str)
data["gage_id"]  = data["gage_id"].astype(str)
pchip["well_id"] = pchip["well_id"].astype(str)

r2_df["slope_sign"] = r2_df["slope"].apply(lambda x: "Negative" if x < 0 else "Positive")

# Short gage labels
gage_short = {
    "BEAR RIVER NEAR CORINNE - UT":                   "Bear River",
    "LITTLE COTTONWOOD CREEK @ JORDAN RIVER NR SLC":  "Little Cottonwood",
    "PROVO RIVER AT PROVO - UT":                       "Provo River",
    "SPANISH FORK NEAR LAKE SHORE - UTAH":             "Spanish Fork",
    "WEBER RIVER NEAR PLAIN CITY - UT":                "Weber River",
}
r2_df["gage_short"] = r2_df["gage_name"].map(gage_short)

# WTE long-term trend
trends = []
for wid, grp in pchip.groupby("well_id"):
    grp = grp.sort_values("date")
    if len(grp) < 24:
        continue
    t = (grp["date"] - grp["date"].min()).dt.days.values.astype(float)
    s, _, r, p, _ = linregress(t, grp["wte"].values)
    trends.append({"well_id": wid, "wte_trend": s * 365, "trend_r2": r**2})
trend_df = pd.DataFrame(trends)

# delta_elev
spatial = (data[["well_id","gage_id","delta_elev","delta_bin"]]
           .drop_duplicates(["well_id","gage_id"]))
r2_merged = (r2_df
             .merge(spatial, on=["well_id","gage_id"], how="left")
             .merge(trend_df, on="well_id", how="left"))

# ── Colours ──────────────────────────────────────────────────────────────────
C = {"Negative": "#E24A33", "Positive": "#4C72B0"}

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Negative vs Positive Slope Wells — Feature Comparison\n(Top-10 R² table)",
             fontsize=14, fontweight="bold", y=0.98)

gs = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.38)

# ── Panel A: per-gage counts (grouped bar) ───────────────────────────────────
ax_a = fig.add_subplot(gs[0, :2])
gage_order = ["Bear River", "Provo River", "Spanish Fork", "Weber River", "Little Cottonwood"]
counts = (r2_df.groupby(["gage_short","slope_sign"])
          .size().unstack(fill_value=0)
          .reindex(gage_order))
x = np.arange(len(gage_order))
w = 0.35
for i, sign in enumerate(["Negative", "Positive"]):
    vals = counts[sign].values if sign in counts.columns else np.zeros(len(gage_order))
    bars = ax_a.bar(x + (i-0.5)*w, vals, w, color=C[sign],
                    label=f"{sign} slope", alpha=0.88, edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 0:
            ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      str(int(v)), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_a.set_xticks(x)
ax_a.set_xticklabels(gage_order, fontsize=9)
ax_a.set_ylabel("Number of wells", fontsize=10)
ax_a.set_title("A  Per-gage distribution of slope sign", fontsize=10, fontweight="bold")
ax_a.legend(fontsize=9)
ax_a.set_ylim(0, counts.values.max() + 2)
ax_a.grid(axis="y", alpha=0.3)

# ── Panel B: R² box ──────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])
data_b = [r2_merged[r2_merged["slope_sign"]==s]["r_squared"].dropna().values
          for s in ["Negative","Positive"]]
bp = ax_b.boxplot(data_b, patch_artist=True, widths=0.5,
                  medianprops=dict(color="white", linewidth=2))
for patch, sign in zip(bp["boxes"], ["Negative","Positive"]):
    patch.set_facecolor(C[sign])
    patch.set_alpha(0.85)
# Overlay points
for i, (vals, sign) in enumerate(zip(data_b, ["Negative","Positive"])):
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
    ax_b.scatter(np.full(len(vals), i+1) + jitter, vals,
                 s=22, color=C[sign], alpha=0.7, zorder=4, edgecolors="none")
u, p = mannwhitneyu(data_b[0], data_b[1], alternative="two-sided")
ax_b.set_xticklabels(["Negative", "Positive"], fontsize=9)
ax_b.set_ylabel("R²", fontsize=10)
ax_b.set_title(f"B  R² distribution\n(MW p={p:.3f})", fontsize=10, fontweight="bold")
ax_b.grid(axis="y", alpha=0.3)

# ── Panel C: n_observations violin ───────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
data_c = [r2_merged[r2_merged["slope_sign"]==s]["n_observations"].dropna().values
          for s in ["Negative","Positive"]]
vp = ax_c.violinplot(data_c, positions=[1,2], showmedians=True,
                     showextrema=True)
for i, (body, sign) in enumerate(zip(vp["bodies"], ["Negative","Positive"])):
    body.set_facecolor(C[sign])
    body.set_alpha(0.7)
vp["cmedians"].set_color("white")
vp["cmedians"].set_linewidth(2)
u, p = mannwhitneyu(data_c[0], data_c[1], alternative="two-sided")
ax_c.set_xticks([1,2])
ax_c.set_xticklabels(["Negative","Positive"], fontsize=9)
ax_c.set_ylabel("n_observations", fontsize=10)
ax_c.set_title(f"C  Observation count\n(MW p={p:.3f})", fontsize=10, fontweight="bold")
ax_c.grid(axis="y", alpha=0.3)

# ── Panel D: |Pearson r| comparison ──────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
data_d = [r2_merged[r2_merged["slope_sign"]==s]["pearson_r"].dropna().values
          for s in ["Negative","Positive"]]
bp2 = ax_d.boxplot(data_d, patch_artist=True, widths=0.5,
                   medianprops=dict(color="white", linewidth=2))
for patch, sign in zip(bp2["boxes"], ["Negative","Positive"]):
    patch.set_facecolor(C[sign])
    patch.set_alpha(0.85)
for i, (vals, sign) in enumerate(zip(data_d, ["Negative","Positive"])):
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
    ax_d.scatter(np.full(len(vals), i+1) + jitter, vals,
                 s=22, color=C[sign], alpha=0.7, zorder=4, edgecolors="none")
ax_d.axhline(0, color="grey", linewidth=0.8, linestyle="--")
u, p = mannwhitneyu(data_d[0], data_d[1], alternative="two-sided")
ax_d.set_xticklabels(["Negative","Positive"], fontsize=9)
ax_d.set_ylabel("Pearson r", fontsize=10)
ax_d.set_title(f"D  Pearson r (ΔQ ~ ΔWTE)\n(MW p={p:.4f})", fontsize=10, fontweight="bold")
ax_d.grid(axis="y", alpha=0.3)

# ── Panel E: WTE long-term trend ─────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])
neg_ids = set(r2_merged[r2_merged["slope_sign"]=="Negative"]["well_id"])
pos_ids = set(r2_merged[r2_merged["slope_sign"]=="Positive"]["well_id"])
data_e = [
    r2_merged[r2_merged["slope_sign"]=="Negative"]["wte_trend"].dropna().values,
    r2_merged[r2_merged["slope_sign"]=="Positive"]["wte_trend"].dropna().values,
]
bp3 = ax_e.boxplot(data_e, patch_artist=True, widths=0.5,
                   medianprops=dict(color="white", linewidth=2))
for patch, sign in zip(bp3["boxes"], ["Negative","Positive"]):
    patch.set_facecolor(C[sign])
    patch.set_alpha(0.85)
for i, (vals, sign) in enumerate(zip(data_e, ["Negative","Positive"])):
    jitter = np.random.default_rng(1).uniform(-0.15, 0.15, len(vals))
    ax_e.scatter(np.full(len(vals), i+1) + jitter, vals,
                 s=22, color=C[sign], alpha=0.7, zorder=4, edgecolors="none")
ax_e.axhline(0, color="grey", linewidth=0.8, linestyle="--", label="No trend")
u, p = mannwhitneyu(data_e[0], data_e[1], alternative="two-sided")
ax_e.set_xticklabels(["Negative","Positive"], fontsize=9)
ax_e.set_ylabel("WTE trend (ft/yr)", fontsize=10)
ax_e.set_title(f"E  WTE long-term trend\n(MW p={p:.3f})", fontsize=10, fontweight="bold")
ax_e.grid(axis="y", alpha=0.3)

# ── Panel F: delta_elev distribution (KDE/histogram overlay) ─────────────────
ax_f = fig.add_subplot(gs[2, 0])
for sign in ["Negative", "Positive"]:
    vals = r2_merged[r2_merged["slope_sign"]==sign]["delta_elev"].dropna().values
    ax_f.hist(vals, bins=15, color=C[sign], alpha=0.55, label=sign, density=True)
ax_f.set_xlabel("delta_elev (ft) = well elevation − reach elevation", fontsize=9)
ax_f.set_ylabel("Density", fontsize=10)
ax_f.set_title("F  Well depth relative to stream", fontsize=10, fontweight="bold")
ax_f.legend(fontsize=9)
ax_f.grid(axis="y", alpha=0.3)
ax_f.axvline(0, color="grey", linewidth=0.8, linestyle="--")

# ── Panel G: delta_bin stacked bar ───────────────────────────────────────────
ax_g = fig.add_subplot(gs[2, 1:])
bin_order = ["< -20", "-20 to -10", "-10 to -5", "-5 to 0",
             "0 to 5", "5 to 10", "10 to 20", "20 to 30"]
bin_counts = (r2_merged.groupby(["slope_sign","delta_bin"])
              .size().unstack(fill_value=0))
# Reindex to ordered bins, keep only those present
present_bins = [b for b in bin_order if b in bin_counts.columns]
bin_counts = bin_counts.reindex(columns=present_bins, fill_value=0)

x_g = np.arange(len(present_bins))
w_g = 0.35
for i, sign in enumerate(["Negative","Positive"]):
    if sign in bin_counts.index:
        vals = bin_counts.loc[sign].values
    else:
        vals = np.zeros(len(present_bins))
    ax_g.bar(x_g + (i-0.5)*w_g, vals, w_g,
             color=C[sign], alpha=0.85, label=f"{sign} slope", edgecolor="white")

ax_g.set_xticks(x_g)
ax_g.set_xticklabels(present_bins, fontsize=8.5, rotation=30, ha="right")
ax_g.set_ylabel("Number of wells", fontsize=10)
ax_g.set_title("G  Elevation bin distribution (well vs. reach)", fontsize=10, fontweight="bold")
ax_g.legend(fontsize=9)
ax_g.grid(axis="y", alpha=0.3)
ax_g.axvline(3.5, color="grey", linewidth=1.0, linestyle="--", alpha=0.6)
ax_g.text(1.5, ax_g.get_ylim()[1]*0.85 if ax_g.get_ylim()[1] > 0 else 5,
          "well below stream", ha="center", fontsize=8, color="grey", style="italic")

# Add legend patches to figure
neg_patch = mpatches.Patch(color=C["Negative"], alpha=0.85, label="Negative slope (ΔQ↓ when ΔWTE↑)")
pos_patch = mpatches.Patch(color=C["Positive"], alpha=0.85, label="Positive slope (ΔQ↑ when ΔWTE↑)")
fig.legend(handles=[neg_patch, pos_patch], loc="lower center",
           ncol=2, fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))

plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {OUT}")
