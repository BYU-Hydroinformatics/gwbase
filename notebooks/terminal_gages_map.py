"""
Recreate the Terminal Gages and Their Upstream Watersheds map for the Great Salt Lake Basin.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATA = BASE / "data"

# ── Load geodata ───────────────────────────────────────────────────────────────
print("Loading geodata...")
basin    = gpd.read_file(DATA / "raw/hydrography/gsl_basin.shp")
catchment = gpd.read_file(DATA / "raw/hydrography/gsl_catchment.shp")
streams  = gpd.read_file(DATA / "raw/hydrography/gslb_stream.shp")
lakes    = gpd.read_file(DATA / "raw/hydrography/lake.shp")
wells    = gpd.read_file(DATA / "raw/hydrography/well_shp.shp")

# ── Load tabular data ──────────────────────────────────────────────────────────
all_gages = pd.read_csv(DATA / "raw/hydrography/gsl_nwm_gage.csv")   # 78 gages
term_df   = pd.read_csv(DATA / "processed/terminal_gages.csv")        # terminal gages
upstream  = pd.read_csv(DATA / "processed/terminal_gage_upstream_catchments.csv")
# columns: Gage_ID, Gage_Name, Terminal_Catchment_ID, Upstream_Catchment_ID

print(f"  Basin polys: {len(basin)}")
print(f"  Catchments:  {len(catchment)}")
print(f"  Terminal gages: {len(term_df)}")
print(f"  All gages:   {len(all_gages)}")
print(f"  Wells:       {len(wells)}")

# ── Reproject everything to Web Mercator for contextily ───────────────────────
TARGET_CRS = "EPSG:3857"
basin     = basin.to_crs(TARGET_CRS)
catchment = catchment.to_crs(TARGET_CRS)
streams   = streams.to_crs(TARGET_CRS)
lakes     = lakes.to_crs(TARGET_CRS)
wells     = wells.to_crs(TARGET_CRS)

# Build GeoDataFrame for all gages
all_gages_gdf = gpd.GeoDataFrame(
    all_gages,
    geometry=gpd.points_from_xy(all_gages["longitude"], all_gages["latitude"]),
    crs="EPSG:4326"
).to_crs(TARGET_CRS)

# Build GeoDataFrame for terminal gages (lat/lon from all_gages)
term_info = term_df.merge(
    all_gages[["id", "latitude", "longitude"]],
    on="id", how="left"
)
term_gdf = gpd.GeoDataFrame(
    term_info,
    geometry=gpd.points_from_xy(term_info["longitude"], term_info["latitude"]),
    crs="EPSG:4326"
).to_crs(TARGET_CRS)

print("Terminal gages:")
for _, r in term_gdf.iterrows():
    print(f"  {r['id']} - {r['name']}")

# ── Assign a distinct color to each terminal gage ────────────────────────────
COLORS = [
    "#E91E8C",  # pink-red      (Bear River / Corinne)
    "#8BC34A",  # lime green    (Weber River)
    "#009688",  # teal          (Farmington Cr)
    "#673AB7",  # deep purple   (Centerville Cr)
    "#1565C0",  # dark blue     (Spanish Fork -- not in term_df, skip)
    "#F57F17",  # amber         (Hobble Creek -- not in term_df, skip)
    "#E91E63",  # magenta       (Provo River -- not in term_df, skip)
    "#CDDC39",  # lime-yellow   (Little Cottonwood)
    "#00BCD4",  # cyan          (Big Cottonwood)
    "#FF5722",  # deep orange   (Vernon Cr)
    "#004D40",  # dark teal     (Warm Cr)
    "#BCAAA4",  # tan           (Dunn Cr)
]

# Use as many colors as terminal gages
term_gdf = term_gdf.reset_index(drop=True)
n_term = len(term_gdf)
colors = COLORS[:n_term]

# ── Build union of upstream catchments per terminal gage ─────────────────────
print("Building upstream watershed polygons...")
watershed_gdfs = []
for i, (_, trow) in enumerate(term_gdf.iterrows()):
    gage_id = trow["id"]
    up_ids  = upstream.loc[upstream["Gage_ID"] == gage_id, "Upstream_Catchment_ID"].astype(int).tolist()
    terminal_catch_id = int(trow["catchment_id"])
    all_ids = set(up_ids) | {terminal_catch_id}

    sub = catchment[catchment["linkno"].astype(int).isin(all_ids)].copy()
    if sub.empty:
        continue
    dissolved = sub.dissolve()
    dissolved["gage_id"]   = gage_id
    dissolved["gage_name"] = trow["name"]
    dissolved["color"]     = colors[i]
    dissolved["idx"]       = i
    watershed_gdfs.append(dissolved)

watershed_gdf = gpd.GeoDataFrame(
    pd.concat(watershed_gdfs, ignore_index=True),
    crs=TARGET_CRS
)
print(f"  Watersheds built: {len(watershed_gdf)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
print("Plotting...")
fig, ax = plt.subplots(figsize=(12, 14))

# 1. Basin boundary (faint outline)
basin.plot(ax=ax, facecolor="none", edgecolor="#aaaaaa", linewidth=0.8, zorder=1)

# 2. Upstream watersheds (colored, semi-transparent)
for _, wrow in watershed_gdf.iterrows():
    gpd.GeoDataFrame([wrow], crs=TARGET_CRS).plot(
        ax=ax,
        facecolor=wrow["color"],
        edgecolor="white",
        linewidth=0.3,
        alpha=0.65,
        zorder=2,
    )

# 3. Streams
streams.plot(ax=ax, color="#1565C0", linewidth=0.4, alpha=0.7, zorder=3)

# 4. Lakes
lakes.plot(ax=ax, facecolor="#90CAF9", edgecolor="#64B5F6", linewidth=0.4, alpha=0.8, zorder=4)

# 5. All gages (orange circles)
all_gages_gdf.plot(ax=ax, color="#FF6D00", markersize=28, marker="o",
                   edgecolor="white", linewidth=0.5, zorder=5)

# 6. Terminal gages (colored stars)
STAR_SIZE = 220
for i, (_, trow) in enumerate(term_gdf.iterrows()):
    ax.scatter(
        trow.geometry.x, trow.geometry.y,
        s=STAR_SIZE, marker="*",
        facecolor=colors[i], edgecolor="black",
        linewidth=0.6, zorder=6
    )

# ── Add basemap ───────────────────────────────────────────────────────────────
try:
    import contextily as ctx
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=8, alpha=0.6)
    print("Basemap added.")
except Exception as e:
    print(f"Basemap skipped: {e}")
    ax.set_facecolor("#f0f0f0")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF6D00",
                  markersize=8, markeredgecolor="white", label=f"All Gages ({len(all_gages)})"),
    mlines.Line2D([0], [0], color="#1565C0", linewidth=1.5, label="Stream Network"),
    mpatches.Patch(facecolor="#90CAF9", edgecolor="#64B5F6", label=f"Lakes ({len(lakes)})"),
    mpatches.Patch(facecolor="#cccccc", edgecolor="none", alpha=0.5, label="Upstream Watersheds"),
]
legend_elements.append(mpatches.Patch(facecolor="none", edgecolor="none", label=""))
legend_elements.append(mpatches.Patch(facecolor="none", edgecolor="none", label=f"Terminal Gages ({n_term}):"))

for i, (_, trow) in enumerate(term_gdf.iterrows()):
    label = f"{trow['id']} - {trow['name'][:35]}..."
    legend_elements.append(
        mlines.Line2D([0], [0], marker="*", color="w",
                      markerfacecolor=colors[i], markersize=12,
                      markeredgecolor="black", markeredgewidth=0.5,
                      label=label)
    )

legend = ax.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=6.5,
    framealpha=0.9,
    edgecolor="#aaaaaa",
    title="Map Elements & Terminal Gages",
    title_fontsize=7.5,
    frameon=True,
    borderpad=0.8,
)

# ── Title ─────────────────────────────────────────────────────────────────────
n_wells = len(wells)
ax.set_title(
    f"Terminal Gages and Their Upstream Watersheds\nGreat Salt Lake Basin\n"
    f"{n_term} Terminal Gages  •  {len(all_gages)} Total Gages  •  {len(lakes)} Lakes  •  {n_wells} Wells",
    fontsize=13, fontweight="bold", pad=12
)

ax.set_axis_off()
plt.tight_layout()

out = BASE / "notebooks/terminal_gages_map.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.show()
