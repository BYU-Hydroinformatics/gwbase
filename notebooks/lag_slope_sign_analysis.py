"""
For Bear River, Weber River, Provo River:
  - Compute per-well slope at each lag (No Lag, 3mo, 6mo, 1yr, 5yr)
  - Show slope distribution per lag
  - Track which negative-at-no-lag wells flip to positive at some lag

Output:
  results/figures/lag_slope_sign/slope_dist_by_lag.png   — distributions
  results/figures/lag_slope_sign/slope_flip_heatmap.png  — per-well sign across lags
  results/features/lag_slope_by_well.csv                 — full table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "figures" / "lag_slope_sign"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_OBS = 20
TARGET_GAGES = {
    "BEAR RIVER NEAR CORINNE - UT":   "Bear River",
    "WEBER RIVER NEAR PLAIN CITY - UT": "Weber River",
    "PROVO RIVER AT PROVO - UT":       "Provo River",
}
TARGET_GAGE_IDS = {"10126000": "Bear River",
                   "10141000": "Weber River",
                   "10163000": "Provo River"}
GAGEID_TO_NAME = {"10126000": "BEAR RIVER NEAR CORINNE - UT",
                  "10141000": "WEBER RIVER NEAR PLAIN CITY - UT",
                  "10163000": "PROVO RIVER AT PROVO - UT"}

# ── Load all lag datasets ────────────────────────────────────────────────────
LAG_SPECS = {
    "No Lag":  ("results/features/data_with_deltas.csv", "delta_wte"),
    "3 Month": ("results/features/data_lag_3mo.csv",     "delta_wte_lag_3_months"),
    "6 Month": ("results/features/data_lag_6mo.csv",     "delta_wte_lag_6_months"),
    "1 Year":  ("results/features/data_lag_1yr.csv",     "delta_wte_lag_1_year"),
    "5 Year":  ("results/features/data_lag_5yr.csv",     "delta_wte_lag_5_years"),
}
LAG_ORDER = ["No Lag", "3 Month", "6 Month", "1 Year", "5 Year"]

def load_lag(path, x_col):
    df = pd.read_csv(BASE / path, low_memory=False)
    df["well_id"] = df["well_id"].astype(str)
    df["gage_id"] = df["gage_id"].astype(str)
    if "gage_name" not in df.columns:
        df["gage_name"] = df["gage_id"].map(GAGEID_TO_NAME)
    df = df[df["gage_id"].isin(TARGET_GAGE_IDS)].copy()
    df["gage_name"] = df["gage_id"].map(GAGEID_TO_NAME)
    return df[["well_id", "gage_id", "gage_name", x_col, "delta_q"]].dropna()

# ── Compute per-well regression at each lag ──────────────────────────────────
records = []
for lag_name in LAG_ORDER:
    path, x_col = LAG_SPECS[lag_name]
    df = load_lag(path, x_col)
    for (well_id, gage_id, gage_name), grp in df.groupby(["well_id","gage_id","gage_name"]):
        x = grp[x_col].values
        y = grp["delta_q"].values
        if len(x) < MIN_OBS or x.std() == 0:
            continue
        s, intercept, r_val, p_val, _ = linregress(x, y)
        records.append({
            "lag": lag_name,
            "well_id": well_id,
            "gage_id": gage_id,
            "gage_name": gage_name,
            "gage_short": TARGET_GAGES[gage_name],
            "slope": s,
            "r_squared": r_val**2,
            "p_value": p_val,
            "n": len(x),
        })

result = pd.DataFrame(records)

# Fix gage_name for files that had placeholder
for lag_name in LAG_ORDER:
    mask = result["lag"] == lag_name
    # already set from gage_name column in data

result.to_csv(RESULTS / "features" / "lag_slope_by_well.csv", index=False)
print(f"Saved lag_slope_by_well.csv  ({len(result)} rows)")

# ── Summary: neg/pos counts per gage per lag ─────────────────────────────────
summary = result.groupby(["gage_short","lag"]).apply(
    lambda g: pd.Series({
        "n_total": len(g),
        "n_neg": (g["slope"] < 0).sum(),
        "n_pos": (g["slope"] > 0).sum(),
        "pct_neg": 100*(g["slope"] < 0).mean(),
        "median_slope": g["slope"].median(),
    }), include_groups=False
).reset_index()
print("\n=== % negative slope per gage per lag ===")
pivot = summary.pivot(index="gage_short", columns="lag", values="pct_neg")[LAG_ORDER]
print(pivot.round(1).to_string())

# ── Fig 1: slope distribution per lag (3 gages × 5 lags) ────────────────────
fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharey=False)
fig.suptitle(
    "Slope Distribution per Lag — Bear River, Weber River, Provo River\n"
    "Red = negative slope,  Blue = positive slope",
    fontsize=13, fontweight="bold"
)

gage_list = ["Bear River", "Weber River", "Provo River"]
for row, gage_short in enumerate(gage_list):
    for col, lag_name in enumerate(LAG_ORDER):
        ax = axes[row, col]
        sub = result[(result["gage_short"]==gage_short) & (result["lag"]==lag_name)]
        if sub.empty:
            ax.set_visible(False)
            continue
        slopes = sub["slope"].values
        n_neg = (slopes < 0).sum()
        n_pos = (slopes > 0).sum()
        n_tot = len(slopes)

        # Clip for display
        p2, p98 = np.percentile(slopes, 2), np.percentile(slopes, 98)
        lo = min(slopes.min(), p2 - abs(p2)*0.05)
        hi = max(slopes.max(), p98 + abs(p98)*0.05)
        slopes_c = np.clip(slopes, lo, hi)

        bins = min(30, max(10, n_tot // 3))
        counts, edges, patches = ax.hist(slopes_c, bins=bins,
                                         edgecolor="white", linewidth=0.3)
        for patch, left in zip(patches, edges[:-1]):
            patch.set_facecolor("#E24A33" if left < 0 else "#4C72B0")
            patch.set_alpha(0.7)

        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", zorder=5)
        ax.set_title(
            f"{lag_name}\n"
            f"neg {n_neg} ({100*n_neg/n_tot:.0f}%)  "
            f"pos {n_pos} ({100*n_pos/n_tot:.0f}%)",
            fontsize=8
        )
        ax.set_xlabel("slope" if row==2 else "", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{gage_short}\nCount", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=7)

plt.tight_layout()
p1 = OUT_DIR / "slope_dist_by_lag.png"
plt.savefig(p1, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved → {p1}")

# ── Fig 2: per-well slope heatmap across lags ────────────────────────────────
# For each gage: rows = wells that are negative at No Lag, cols = lags
# Cell colour = slope value (diverging, white=0)

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 10),
                            gridspec_kw={"wspace": 0.45})
fig2.suptitle(
    "Per-well slope across lags\n"
    "Wells sorted by No-Lag slope  |  Red < 0 < Blue",
    fontsize=12, fontweight="bold"
)

for col, gage_short in enumerate(gage_list):
    ax = axes2[col]
    sub = result[result["gage_short"] == gage_short]
    # pivot: rows = well_id, cols = lag
    piv = sub.pivot_table(index="well_id", columns="lag",
                          values="slope", aggfunc="first")
    piv = piv.reindex(columns=LAG_ORDER)
    # Sort by No Lag slope
    if "No Lag" in piv.columns:
        piv = piv.sort_values("No Lag")

    # Clip colour scale at ±50 for readability
    vmax = min(50, piv.abs().quantile(0.95).max())
    vmin = -vmax

    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu",
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    # Overlay text for sign
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=5, color="#AAAAAA")
            else:
                sign = "▼" if val < 0 else "▲"
                ax.text(j, i, sign, ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(val) > vmax*0.5 else "black")

    ax.set_xticks(range(len(LAG_ORDER)))
    ax.set_xticklabels(LAG_ORDER, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=5)
    ax.set_title(f"{gage_short}  (n={len(piv)} wells)", fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="slope (cfs/ft)")

    # Annotate flip count
    if "No Lag" in piv.columns:
        neg_nolag = piv[piv["No Lag"] < 0]
        flip_counts = {}
        for lag in LAG_ORDER[1:]:
            if lag in piv.columns:
                flipped = (neg_nolag[lag] > 0).sum()
                total   = neg_nolag[lag].notna().sum()
                flip_counts[lag] = (flipped, total)
        flip_str = "Neg→Pos flips (from No-Lag neg wells):\n" + "\n".join(
            f"  {k}: {v[0]}/{v[1]}" for k, v in flip_counts.items()
        )
        ax.text(1.02, 0.02, flip_str,
                transform=ax.transAxes, fontsize=7.5,
                va="bottom", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0",
                          edgecolor="#CCCCCC"))

p2 = OUT_DIR / "slope_flip_heatmap.png"
plt.savefig(p2, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved → {p2}")

# ── Fig 3: % negative slope vs lag (line chart) ──────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))
colors3 = {"Bear River": "#E24A33", "Weber River": "#4C72B0", "Provo River": "#2CA02C"}
for gage_short, color in colors3.items():
    row_data = summary[summary["gage_short"]==gage_short].set_index("lag")["pct_neg"]
    row_data = row_data.reindex(LAG_ORDER)
    ax3.plot(LAG_ORDER, row_data.values, marker="o", linewidth=2,
             color=color, label=gage_short, markersize=8)
    for x, y in zip(LAG_ORDER, row_data.values):
        if not np.isnan(y):
            ax3.text(x, y+1.5, f"{y:.0f}%", ha="center", fontsize=8, color=color)

ax3.axhline(50, color="grey", linewidth=1, linestyle="--", alpha=0.6, label="50% threshold")
ax3.set_ylabel("% wells with negative slope", fontsize=11)
ax3.set_xlabel("Lag", fontsize=11)
ax3.set_title("Does adding lag reduce negative-slope prevalence?", fontsize=12, fontweight="bold")
ax3.legend(fontsize=10)
ax3.set_ylim(0, 100)
ax3.grid(alpha=0.3)

p3 = OUT_DIR / "pct_negative_vs_lag.png"
plt.savefig(p3, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print(f"Saved → {p3}")

print("\nDone.")
