"""
For every well in top10_by_r2.csv and top10_by_mi.csv where slope < 0,
plot the original raw time series and mark the monthly PCHIP-interpolated points.

Output:
  results/figures/negative_slope_timeseries/  — one PNG per well (named by well_id)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

BASE    = Path(__file__).parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "figures" / "negative_slope_timeseries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load tables ──────────────────────────────────────────────────────────────
r2_df = pd.read_csv(RESULTS / "figures" / "top10_wells_scatter" / "tables" / "top10_by_r2.csv")
mi_df = pd.read_csv(RESULTS / "figures" / "top10_wells_scatter" / "tables" / "top10_by_mi.csv")

neg_r2 = r2_df[r2_df["slope"] < 0][["gage_id", "gage_name", "well_id", "slope", "r_squared"]].copy()
neg_mi = mi_df[mi_df["slope_reg"] < 0][["gage_id", "gage_name", "well_id", "slope_reg", "mi"]].copy()
neg_mi = neg_mi.rename(columns={"slope_reg": "slope"})

# Combine: keep one row per well (prefer R2 entry if duplicate)
combined = pd.concat([
    neg_r2.assign(source="R²"),
    neg_mi.assign(source="MI")
], ignore_index=True)
combined["well_id"] = combined["well_id"].astype(str)
# If a well appears in both, keep both rows (separate annotations)
# But only plot once — deduplicate by well_id, keeping first occurrence
well_meta = (
    combined.drop_duplicates("well_id")
    .set_index("well_id")
    [["gage_id", "gage_name", "slope"]]
)
# Carry all rows to build a per-well annotation string
def build_annotation(well_id):
    rows = combined[combined["well_id"] == well_id]
    parts = []
    for _, row in rows.iterrows():
        if row["source"] == "R²":
            parts.append(f"R² table: slope={row['slope']:.3f}, R²={row['r_squared']:.3f}")
        else:
            parts.append(f"MI table:  slope={row['slope']:.3f}, MI={row['mi']:.3f}")
    return "\n".join(parts)

# ── Load raw + PCHIP data ────────────────────────────────────────────────────
raw   = pd.read_csv(RESULTS / "processed" / "well_ts_cleaned.csv",
                    usecols=["well_id", "date", "wte", "is_outlier"],
                    parse_dates=["date"])
pchip = pd.read_csv(RESULTS / "processed" / "well_pchip_monthly.csv",
                    usecols=["well_id", "date", "wte"],
                    parse_dates=["date"])
raw["well_id"]   = raw["well_id"].astype(str)
pchip["well_id"] = pchip["well_id"].astype(str)

# ── Plot ─────────────────────────────────────────────────────────────────────
unique_wells = list(well_meta.index)
print(f"Wells with negative slope: {len(unique_wells)}")

for well_id in unique_wells:
    meta      = well_meta.loc[well_id]
    gage_id   = meta["gage_id"]
    gage_name = meta["gage_name"]
    annot     = build_annotation(well_id)

    w_raw   = raw[raw["well_id"] == well_id].sort_values("date")
    w_pchip = pchip[pchip["well_id"] == well_id].sort_values("date")

    if w_raw.empty and w_pchip.empty:
        print(f"  {well_id}: no data – skipping.")
        continue

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Outlier mask
    good    = w_raw[~w_raw["is_outlier"]]
    outlier = w_raw[w_raw["is_outlier"]]

    # PCHIP line (drawn first so raw points sit on top)
    if not w_pchip.empty:
        ax.plot(w_pchip["date"], w_pchip["wte"],
                color="#4C72B0", linewidth=1.0, alpha=0.55,
                label="PCHIP monthly (interpolated)", zorder=2)
        ax.scatter(w_pchip["date"], w_pchip["wte"],
                   s=14, color="#4C72B0", alpha=0.75,
                   edgecolors="none", zorder=3,
                   label="_nolegend_")

    # Raw good measurements
    if not good.empty:
        ax.scatter(good["date"], good["wte"],
                   s=35, color="#E24A33", zorder=5,
                   edgecolors="white", linewidths=0.5,
                   label="Raw measurements")

    # Outliers (flagged)
    if not outlier.empty:
        ax.scatter(outlier["date"], outlier["wte"],
                   s=35, color="grey", zorder=4, marker="x",
                   label="Outlier (flagged)")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("WTE (ft)", fontsize=10)
    ax.set_title(
        f"Well {well_id}  —  Gage {gage_id}  ({gage_name})\n"
        f"Raw measurements + PCHIP monthly interpolation",
        fontsize=10, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.25)

    ax.text(0.01, 0.02, annot,
            transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="#AAAAAA"))

    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    plt.tight_layout()
    out_path = OUT_DIR / f"{well_id}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")

print(f"\nDone. Outputs → {OUT_DIR}")
