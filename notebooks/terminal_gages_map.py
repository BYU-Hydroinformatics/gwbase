"""
Terminal Gages and Their Upstream Watersheds
Plotting code ported from create_enhanced_watershed_visualization(),
with gsl_lake.shp replacing the old lake.shp.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke, Normal
from pathlib import Path

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

major_streams = stream_gdf[stream_gdf['strmOrder'] >= 4].copy()
print(f"  Streams total: {len(stream_gdf)}, order >= 4: {len(major_streams)}")

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
               label=f'Groundwater Wells (n={len(well_in_basin)})'),
    # blank spacer
    mpatches.Patch(facecolor='none', edgecolor='none', label=''),
    mpatches.Patch(facecolor='none', edgecolor='none',
                   label=f'Terminal Gages ({len(terminal_gage_info)}):'),
]
for entry in terminal_gage_info:
    name_trunc = entry['name'][:38] + '...' if len(entry['name']) > 38 else entry['name']
    legend_elements.append(
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor=entry['color'], markersize=12,
                   markeredgecolor='black', markeredgewidth=0.5,
                   label=f"{entry['id']} - {name_trunc}")
    )

legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=6.5,
                   title='Map Elements & Terminal Gages', title_fontsize=7.5,
                   frameon=True, framealpha=0.9,
                   facecolor='white', edgecolor='#aaaaaa', labelcolor='#1E293B',
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
