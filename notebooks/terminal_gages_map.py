"""
Terminal Gages and Their Upstream Watersheds
Plotting code ported from create_enhanced_watershed_visualization(),
with gsl_lake.shp replacing the old lake.shp.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import shapely.geometry as sgeom
from matplotlib.patheffects import withStroke, Normal
from pathlib import Path


def filter_major_streams(stream_gdf, min_order=4, min_component_km=50):
    """
    Select the "major streams" to draw on this map.

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
    major = stream_gdf[stream_gdf['strmOrder'] >= min_order].copy()

    linknos = set(major['LINKNO'])
    graph = nx.Graph()
    graph.add_nodes_from(linknos)
    for _, row in major.iterrows():
        if row['DSLINKNO'] in linknos:
            graph.add_edge(row['LINKNO'], row['DSLINKNO'])
    length_km = dict(zip(major['LINKNO'], major.to_crs(3857).geometry.length / 1000))
    keep = set()
    for component in nx.connected_components(graph):
        if sum(length_km.get(l, 0) for l in component) >= min_component_km:
            keep |= component
    major = major[major['LINKNO'].isin(keep)]

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


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
DATA    = BASE / "data"
RESULTS = BASE / "results"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
terminal_mapping = pd.read_csv(RESULTS / "processed" / "terminal_gage_upstream_catchments.csv")
subbasin_gdf     = gpd.read_file(DATA / "raw/hydrography/gsl_catchment.shp")
gage_df          = pd.read_csv(DATA / "raw/hydrography/gsl_nwm_gage.csv")
well_gdf         = gpd.read_file(DATA / "raw/hydrography/well_shp.shp")
stream_gdf       = gpd.read_file(DATA / "raw/hydrography/gslb_stream.shp")
lake_gdf         = gpd.read_file(DATA / "raw/hydrography/gsl_lake.shp")

# ── Mask lake-crossing GEOGLOWS routing artifacts (R2 comment 3) ─────────────
# GEOGLOWS routes reaches *through* lake/reservoir polygons to preserve
# upstream-downstream network topology across standing water bodies, so
# reaches entirely contained within the lake are a synthetic connectivity
# mesh, not real channels. Reaches that merely clip the lake's edge (e.g.
# Bear River's real outlet into Bear River Bay) are kept. Reuses the same
# lake-intersection test verified in review1/revision_calcs/
# 05_lake_reach_verification/ (notebooks/round1_lake_reach_verification.py).
stream_gdf['LINKNO'] = stream_gdf['LINKNO'].astype('int64')
stream_gdf['DSLINKNO'] = stream_gdf['DSLINKNO'].astype('int64')
lake_for_mask = lake_gdf.to_crs(stream_gdf.crs)
lake_union = lake_for_mask.union_all() if hasattr(lake_for_mask, "union_all") else lake_for_mask.unary_union
stream_gdf = stream_gdf[~stream_gdf.geometry.within(lake_union)].copy()

major_streams = filter_major_streams(stream_gdf)

print(f"  Streams total: {len(stream_gdf)}, order >= 4 (filtered): {len(major_streams)}")

# ── Preprocessing ──────────────────────────────────────────────────────────────
linkno_col = 'linkno' if 'linkno' in subbasin_gdf.columns else 'LINKNO'

if 'Gage_ID' in terminal_mapping.columns:
    terminal_mapping = terminal_mapping.rename(columns={
        'Gage_ID': 'gage_id',
        'Upstream_Catchment_ID': 'upstream_catchment_id'
    })

terminal_mapping = terminal_mapping.dropna(subset=['upstream_catchment_id'])
terminal_mapping['gage_id'] = terminal_mapping['gage_id'].astype(int)
terminal_mapping['upstream_catchment_id'] = terminal_mapping['upstream_catchment_id'].astype(int)

gage_df['id'] = gage_df['id'].astype(int)
subbasin_gdf = subbasin_gdf.dropna(subset=[linkno_col])
subbasin_gdf[linkno_col] = subbasin_gdf[linkno_col].astype(int)

terminal_gage_ids = terminal_mapping['gage_id'].unique().tolist()
terminal_gages    = gage_df[gage_df['id'].isin(terminal_gage_ids)].copy()

available_catchments = set(subbasin_gdf[linkno_col].unique())
terminal_gage_catchments = {}
for gage_id in terminal_gage_ids:
    up = terminal_mapping[terminal_mapping['gage_id'] == gage_id]['upstream_catchment_id'].tolist()
    valid = [c for c in up if c in available_catchments]
    if valid:
        terminal_gage_catchments[gage_id] = set(valid)

# Clip wells to basin
subbasin_union = subbasin_gdf.dissolve()
well_gdf_proj  = well_gdf.to_crs(subbasin_gdf.crs)
well_in_basin  = gpd.sjoin(
    well_gdf_proj, subbasin_union[['geometry']],
    how='inner', predicate='within'
).drop(columns=['index_right'])
print(f"  Wells total: {len(well_gdf)}, within subbasin: {len(well_in_basin)}")

# ── Reproject to Web Mercator ──────────────────────────────────────────────────
subbasin_web      = subbasin_gdf.to_crs('EPSG:3857')
major_streams_web = major_streams.to_crs('EPSG:3857')
lake_web          = lake_gdf.to_crs('EPSG:3857')
well_web          = well_in_basin.to_crs('EPSG:3857')

terminal_gages_web = gpd.GeoDataFrame(
    terminal_gages,
    geometry=gpd.points_from_xy(terminal_gages['longitude'], terminal_gages['latitude']),
    crs='EPSG:4326'
).to_crs('EPSG:3857')

# ── Colors ─────────────────────────────────────────────────────────────────────
bright_vivid_colors = [
    '#EF4444', '#10B981', '#3B82F6', '#FBBF24',
    '#8B5CF6', '#06B6D4', '#F59E0B', '#EC4899',
    '#14B8A6', '#A855F7', '#6366F1', '#84CC16',
]
terminal_gage_colors = dict(zip(terminal_gage_ids, bright_vivid_colors[:len(terminal_gage_ids)]))

BG_COLOR     = 'white'
BASIN_FILL   = '#FAFBFC'
BASIN_EDGE   = '#E8ECF0'
STREAM_COLOR = '#0369A1'
LAKE_COLOR   = '#38BDF8'
WELL_COLOR   = '#BE185D'
OUTLINE_COLOR = '#475569'

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(22, 16))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# 1. Subbasin background
subbasin_web.plot(ax=ax, color=BASIN_FILL, edgecolor=BASIN_EDGE,
                  linewidth=0.2, alpha=1.0, zorder=1)

# 2. Colored upstream watersheds
for gage_id in terminal_gage_ids:
    if gage_id not in terminal_gage_catchments:
        continue
    upstream_basins = subbasin_web[
        subbasin_web[linkno_col].isin(list(terminal_gage_catchments[gage_id]))
    ]
    if not upstream_basins.empty:
        upstream_basins.plot(ax=ax, color=terminal_gage_colors[gage_id],
                             alpha=0.75, edgecolor='none', zorder=2)

# Basin outer boundary
subbasin_web.dissolve().boundary.plot(ax=ax, color=OUTLINE_COLOR,
                                      linewidth=2.2, alpha=0.85, zorder=2.5)

# 3. Lakes
lake_web.plot(ax=ax, color=LAKE_COLOR, edgecolor='#38BDF8',
              linewidth=0.5, alpha=0.6, zorder=3)

# 4. Streams — width hierarchy by Strahler order
ms = major_streams_web.copy()
ms['strmOrder'] = pd.to_numeric(ms['strmOrder'], errors='coerce')
ms['__lw__'] = 1.5 + 0.6 * (ms['strmOrder'] - 4).clip(lower=0)
for lw_val, grp in ms.groupby('__lw__'):
    grp.plot(ax=ax, color=STREAM_COLOR, linewidth=float(lw_val), alpha=0.9, zorder=4)

# 5. Wells
well_web.plot(ax=ax, marker='o', markersize=8, color=WELL_COLOR,
              edgecolor='none', alpha=0.80, zorder=5, rasterized=True)

# 6. Terminal gage stars
terminal_gage_info = []
for _, row in terminal_gages_web.iterrows():
    gage_id = int(row['id'])
    if gage_id not in terminal_gage_catchments:
        continue
    c = terminal_gage_colors[gage_id]
    star = ax.scatter([row.geometry.x], [row.geometry.y],
                      c=c, marker='*', s=600,
                      edgecolors='none', linewidths=0, alpha=1.0, zorder=10)
    star.set_path_effects([
        withStroke(linewidth=8,  foreground='#000000', alpha=0.35),
        withStroke(linewidth=5,  foreground='#000000', alpha=0.55),
        withStroke(linewidth=2.5, foreground='white',  alpha=1.0),
        Normal()
    ])
    terminal_gage_info.append({'id': gage_id, 'name': row.get('name', f'Gage {gage_id}'), 'color': c})

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_elements = [
    plt.Line2D([0], [0], color=STREAM_COLOR, linewidth=3.0, label='Major Streams'),
    mpatches.Patch(facecolor=LAKE_COLOR, edgecolor='#38BDF8', alpha=0.6, label='Lakes'),
    mpatches.Patch(facecolor='#94A3B8', edgecolor='none', alpha=0.65, label='Upstream Watersheds'),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=WELL_COLOR,
               markeredgecolor='none', markersize=9,
               label='Groundwater Wells'),
    # blank spacer
    mpatches.Patch(facecolor='none', edgecolor='none', label=''),
    mpatches.Patch(facecolor='none', edgecolor='none', label='Terminal Gages:'),
]
for entry in terminal_gage_info:
    name_trunc = entry['name'][:38] + '...' if len(entry['name']) > 38 else entry['name']
    legend_elements.append(
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor=entry['color'], markersize=12,
                   markeredgecolor='black', markeredgewidth=0.5,
                   label=f"{entry['id']} - {name_trunc}")
    )

legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
                   title='Map Elements & Terminal Gages', title_fontsize=12,
                   frameon=True, framealpha=0.95,
                   facecolor='white', edgecolor='#475569', labelcolor='#1E293B',
                   borderpad=0.8)
legend.get_title().set_color('#1E293B')
legend.get_title().set_fontweight('bold')

ax.set_aspect('equal')
ax.axis('off')
ax.margins(0.01)
plt.tight_layout()

out = RESULTS / "analysis" / "maps" / "overview" / "terminal_gages_map.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=600, bbox_inches='tight', facecolor=BG_COLOR, edgecolor='none')
print(f"Saved → {out}")
plt.show()
