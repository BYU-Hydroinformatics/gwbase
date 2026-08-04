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
import networkx as nx
import shapely.geometry as sgeom


def filter_major_streams(stream_gdf, min_order=4, min_component_km=50):
    """
    Select the "major streams" to draw on a basin overview map.

    Two problems with a naive `strmOrder >= min_order` cutoff, both found by
    inspecting the rendered map rather than assumed up front:

    1. A hard order cutoff strands reaches whose immediate up/downstream
       neighbors fall below the cutoff, breaking otherwise-continuous rivers
       into disconnected-looking pieces. Grouping by the GEOGLOWS
       LINKNO/DSLINKNO attribute chain and keeping only components with at
       least `min_component_km` of total length recovers the ~9 major named
       river systems in this basin.
    2. That attribute chain is not fully reliable: a handful of reaches
       still render as isolated squiggles because DSLINKNO links them to a
       large component they don't actually touch on the ground. These are
       removed by explicit bounding box below. Coordinates come from
       querying each piece's exact geometric bounds directly, not from
       eyeballing a rendered image.

    Caveat: the box list is tied to this exact GEOGLOWS shapefile snapshot.
    If the source data changes, re-derive them rather than assuming they
    still apply.
    """
    major = stream_gdf[stream_gdf["strmOrder"] >= min_order].copy()

    linknos = set(major["LINKNO"])
    graph = nx.Graph()
    graph.add_nodes_from(linknos)
    for _, row in major.iterrows():
        if row["DSLINKNO"] in linknos:
            graph.add_edge(row["LINKNO"], row["DSLINKNO"])
    length_km = dict(zip(major["LINKNO"], major.to_crs(3857).geometry.length / 1000))
    keep = set()
    for component in nx.connected_components(graph):
        if sum(length_km.get(l, 0) for l in component) >= min_component_km:
            keep |= component
    major = major[major["LINKNO"].isin(keep)]

    isolated_boxes = [
        (-112.45, 40.14, -112.24, 40.46),  # ~77.5 km piece, lat 40.2-40.4
        (-113.54, 39.14, -113.44, 39.41),  # ~55.5 km piece, lat 39.2-39.4
        (-113.74, 38.26, -113.64, 38.65),  # ~81.4 km piece, lat 38.3-38.6
        (-112.25, 40.16, -112.21, 40.20),  # 3-reach sliver of an 89 km component
    ]
    for minx, miny, maxx, maxy in isolated_boxes:
        box = sgeom.box(minx, miny, maxx, maxy)
        major = major[~major.geometry.intersects(box)]

    return major


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

# ── Mask lake-crossing GEOGLOWS routing artifacts (R2 comment 3) ─────────────
# GEOGLOWS routes reaches *through* lake/reservoir polygons to preserve
# upstream-downstream network topology across standing water bodies, so
# reaches entirely contained within the lake are a synthetic connectivity
# mesh, not real channels. Reaches that merely clip the lake's edge (e.g.
# Bear River's real outlet into Bear River Bay) are kept. Reuses the same
# lake-intersection test verified in review1/revision_calcs/
# 05_lake_reach_verification/ (notebooks/round1_lake_reach_verification.py).
gsl_stream["LINKNO"] = gsl_stream["LINKNO"].astype("int64")
gsl_stream["DSLINKNO"] = gsl_stream["DSLINKNO"].astype("int64")
lake_gdf = lakes.to_crs(gsl_stream.crs)
lake_union = lake_gdf.union_all() if hasattr(lake_gdf, "union_all") else lake_gdf.unary_union
gsl_stream = gsl_stream[~gsl_stream.geometry.within(lake_union)].copy()

major_streams = filter_major_streams(gsl_stream)

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
plt.savefig(out1, dpi=600, bbox_inches="tight", facecolor="white")
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
plt.savefig(out2, dpi=600, bbox_inches="tight", facecolor="white")
print(f"Saved → {out2}")
plt.show()
