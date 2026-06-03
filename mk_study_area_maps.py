"""
Great Salt Lake Basin Study Area Maps
  Map 1: Groundwater Wells
  Map 2: Major Streams, Catchments, and Stream Gages
"""

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
HYD  = BASE / "data" / "raw" / "hydrography"
OUT  = BASE / "results" / "analysis" / "maps" / "overview"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading shapefiles...")
basin      = gpd.read_file(HYD / "gsl_basin.shp")
gsl_stream = gpd.read_file(HYD / "gslb_stream.shp")
lakes      = gpd.read_file(HYD / "gsl_lake.shp")
catches    = gpd.read_file(HYD / "gsl_catchment.shp")
wells      = gpd.read_file(HYD / "well_shp.shp")
gages_df   = pd.read_csv(HYD / "gsl_nwm_gage.csv").dropna(subset=["longitude", "latitude"])

major_streams = gsl_stream[gsl_stream["strmOrder"] >= 4]

# Reproject all to basin CRS
target_crs = basin.crs
for gdf in [gsl_stream, lakes, catches, wells]:
    gdf.to_crs(target_crs, inplace=True)

# ── Shared style constants ────────────────────────────────────────────────────
BASIN_COLOR  = "#F8F9FA"
BASIN_EDGE   = "#2C3E50"
STREAM_COLOR = "#1E88E5"
CATCH_EDGE   = "#BDC3C7"
LAKE_COLOR   = "#87CEEB"
WELL_COLOR   = "#E74C3C"
GAGE_COLOR   = "#E74C3C"
BG_COLOR     = "#F7F9FC"
BBOX_PROPS   = dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.95,
                    edgecolor=BASIN_EDGE, linewidth=2)


# ══════════════════════════════════════════════════════════════════════════════
# Map 1 — Groundwater Wells
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(10, 11))

basin.plot(ax=ax, color=BASIN_COLOR, alpha=0.8, edgecolor=BASIN_EDGE, linewidth=2.0)
lakes.plot(ax=ax, color=LAKE_COLOR, alpha=0.4)
wells.plot(ax=ax, color=WELL_COLOR, markersize=4, alpha=0.9)

ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_aspect("equal", adjustable="box")
ax.set_facecolor(BG_COLOR)
ax.tick_params(axis="both", which="major", labelsize=12)

ax.text(0.02, 0.98, f"Wells: {len(wells):,}", transform=ax.transAxes, fontsize=12,
        verticalalignment="top", bbox=BBOX_PROPS, family="monospace")

legend_elements = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=WELL_COLOR,
               markersize=10, label="Wells", linestyle="None"),
    mpatches.Patch(facecolor=LAKE_COLOR, edgecolor="none", label="Lakes"),
    plt.Line2D([0], [0], color=BASIN_EDGE, linewidth=4, label="Basin Boundary"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=13,
          frameon=True, fancybox=True, shadow=True)

plt.tight_layout(pad=2.0)
out1 = OUT / "gslb_wells_map.png"
plt.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out1}")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Map 2 — Major Streams, Catchments, and Stream Gages
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(10, 11))

basin.plot(ax=ax, color=BASIN_COLOR, alpha=0.8, edgecolor=BASIN_EDGE, linewidth=2.0)
catches.plot(ax=ax, color="none", edgecolor=CATCH_EDGE, linewidth=0.3, alpha=0.6)
major_streams.plot(ax=ax, color=STREAM_COLOR, linewidth=1.5, alpha=0.9)
lakes.plot(ax=ax, color=LAKE_COLOR, alpha=0.4)
ax.scatter(gages_df["longitude"], gages_df["latitude"],
           color=GAGE_COLOR, s=100, alpha=0.9,
           edgecolors="black", linewidths=1, zorder=5)

ax.set_xlabel("Longitude", fontsize=14)
ax.set_ylabel("Latitude", fontsize=14)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_aspect("equal", adjustable="box")
ax.set_facecolor(BG_COLOR)
ax.tick_params(axis="both", which="major", labelsize=12)

info_text = (
    f"Catchments: {len(catches):,}\n"
    f"Total Streams: {len(gsl_stream):,}\n"
    f"Stream Gages: {len(gages_df)}"
)
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
        verticalalignment="top", bbox=BBOX_PROPS, family="monospace")

legend_elements = [
    plt.Line2D([0], [0], color=STREAM_COLOR, linewidth=4, label="Major Streams"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GAGE_COLOR,
               markeredgecolor="black", markersize=10, label="Stream Gages", linestyle="None"),
    mpatches.Patch(facecolor=LAKE_COLOR, edgecolor="none", label="Lakes"),
    plt.Line2D([0], [0], color=CATCH_EDGE, linewidth=3, label="Catchments"),
    plt.Line2D([0], [0], color=BASIN_EDGE, linewidth=4, label="Basin Boundary"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=13,
          frameon=True, fancybox=True, shadow=True)

plt.tight_layout(pad=2.0)
out2 = OUT / "gslb_streams_gages_map.png"
plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out2}")
plt.show()
