"""
Map for terminal gage 10153100 (HOBBLE CREEK AT 1650 WEST AT SPRINGVILLE)
with well 400938111365801 highlighted.
Standalone script – does not modify the batch map output.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATA = BASE / "data"
OUT  = BASE / "results" / "figures" / "terminal_gage_maps"
OUT.mkdir(parents=True, exist_ok=True)

GAGE_ID       = 10153100
HIGHLIGHT_WELL_ID = 400938111365801   # Well_ID as integer

# ── Colors (same palette as batch maps) ───────────────────────────────────────
COL_UPSTREAM  = "#C7E9C0"
COL_UP_EDGE   = "#74C476"
COL_TERMINAL  = "#FDAE6B"
COL_TERM_EDGE = "#D94801"
COL_STREAM    = "#2171B5"
COL_LAKE      = "#9ECAE1"
COL_LAKE_EDGE = "#6BAED6"
COL_WELL      = "#8B4513"
COL_GAGE      = "#E31A1C"
COL_BASIN_FILL = "#EEEEEE"
COL_BASIN_EDGE = "#888888"
COL_HL_WELL   = "#FFD700"   # gold highlight for the target well
COL_HL_EDGE   = "#B8860B"

TARGET_CRS = "EPSG:3857"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading geodata...")
basin     = gpd.read_file(DATA / "raw/hydrography/gsl_basin.shp").to_crs(TARGET_CRS)
catchment = gpd.read_file(DATA / "raw/hydrography/gsl_catchment.shp").to_crs(TARGET_CRS)
streams   = gpd.read_file(DATA / "raw/hydrography/gslb_stream.shp").to_crs(TARGET_CRS)
lakes     = gpd.read_file(DATA / "raw/hydrography/lake.shp")          # keep 4326 for GSL lookup
lakes_3857 = lakes.to_crs(TARGET_CRS)
wells_gdf = gpd.read_file(DATA / "raw/hydrography/well_shp.shp").to_crs(TARGET_CRS)

all_gages = pd.read_csv(DATA / "raw/hydrography/gsl_nwm_gage.csv")
RESULTS = BASE / "results"
term_df   = pd.read_csv(RESULTS / "processed/terminal_gages.csv")
upstream  = pd.read_csv(RESULTS / "processed/terminal_gage_upstream_catchments.csv")

catchment["linkno_int"] = catchment["linkno"].astype(float).astype(int)

# ── Terminal gage row ──────────────────────────────────────────────────────────
trow_df = term_df[term_df["id"] == GAGE_ID].merge(
    all_gages[["id", "latitude", "longitude", "name"]],
    on="id", how="left", suffixes=("", "_gage"),
)
trow_df["display_name"] = trow_df["name"].fillna(trow_df.get("name_gage", ""))
trow_gdf = gpd.GeoDataFrame(
    trow_df,
    geometry=gpd.points_from_xy(trow_df["longitude"], trow_df["latitude"]),
    crs="EPSG:4326",
).to_crs(TARGET_CRS)

assert len(trow_gdf) == 1, f"Expected 1 row for gage {GAGE_ID}, got {len(trow_gdf)}"
trow = trow_gdf.iloc[0]
gage_name = trow["display_name"]
print(f"Gage: {GAGE_ID} – {gage_name}")

# ── Catchment IDs ─────────────────────────────────────────────────────────────
terminal_catch_id = int(float(trow["catchment_id"]))
up_ids = (
    upstream.loc[upstream["Gage_ID"] == GAGE_ID, "Upstream_Catchment_ID"]
    .astype(float).astype(int).tolist()
)
all_catch_ids     = set(up_ids) | {terminal_catch_id}
upstream_only_ids = set(up_ids) - {terminal_catch_id}

terminal_catch = catchment[catchment["linkno_int"] == terminal_catch_id].copy()
upstream_catch = catchment[catchment["linkno_int"].isin(upstream_only_ids)].copy()
all_catch      = catchment[catchment["linkno_int"].isin(all_catch_ids)].copy()

total_area = all_catch.dissolve()
total_geom = total_area.geometry.iloc[0]

streams_clip = streams[streams.intersects(total_geom)].copy()
wells_clip   = wells_gdf[wells_gdf.within(total_geom)].copy()
lakes_clip   = lakes_3857[lakes_3857.intersects(total_geom)].copy()

# ── Identify highlighted well ──────────────────────────────────────────────────
# Well_ID is stored as float in the shapefile
highlight_well = wells_gdf[
    wells_gdf["Well_ID"].astype(float).astype(int) == HIGHLIGHT_WELL_ID
].copy()
print(f"Highlighted well rows found: {len(highlight_well)}")
if highlight_well.empty:
    raise ValueError(f"Well {HIGHLIGHT_WELL_ID} not found in shapefile.")
hl_row = highlight_well.iloc[0]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    width_ratios=[0.72, 0.28],
    height_ratios=[0.50, 0.50],
    hspace=0.05, wspace=0.02,
    left=0.01, right=0.99,
    top=0.92, bottom=0.04,
)
ax     = fig.add_subplot(gs[:, 0])
ax_ins = fig.add_subplot(gs[0, 1])
ax_leg = fig.add_subplot(gs[1, 1])

# ── Main map ───────────────────────────────────────────────────────────────────
# 1. Lakes (below catchments)
if not lakes_clip.empty:
    lakes_clip.plot(ax=ax, facecolor=COL_LAKE, edgecolor=COL_LAKE_EDGE,
                    linewidth=0.4, alpha=0.75, zorder=1)

# 2. Upstream catchments
if not upstream_catch.empty:
    upstream_catch.plot(ax=ax, facecolor=COL_UPSTREAM, edgecolor=COL_UP_EDGE,
                        linewidth=0.3, alpha=0.75, zorder=2)

# 3. Terminal catchment
if not terminal_catch.empty:
    terminal_catch.plot(ax=ax, facecolor=COL_TERMINAL, edgecolor=COL_TERM_EDGE,
                        linewidth=1.5, alpha=0.90, zorder=3)

# 4. Watershed boundary
total_area.plot(ax=ax, facecolor="none", edgecolor="#1A3A5C",
                linewidth=1.2, zorder=4)

# 5. Streams
if not streams_clip.empty:
    streams_clip.plot(ax=ax, color=COL_STREAM, linewidth=0.7, alpha=0.85, zorder=5)

# 6. Regular wells (excluding highlighted one)
wells_other = wells_clip[
    wells_clip["Well_ID"].astype(float).astype(int) != HIGHLIGHT_WELL_ID
]
if not wells_other.empty:
    wells_other.plot(ax=ax, color=COL_WELL, markersize=18, marker="o",
                     alpha=0.75, edgecolor="white", linewidth=0.5, zorder=7)

# 7. Highlighted well (gold, same marker as other wells but different color)
ax.scatter(
    hl_row.geometry.x, hl_row.geometry.y,
    s=18, marker="o",
    facecolor=COL_HL_WELL, edgecolor=COL_HL_EDGE,
    linewidth=0.8, zorder=11,
)

# 8. Terminal gage
ax.scatter(
    trow.geometry.x, trow.geometry.y,
    s=600, marker="*",
    facecolor=COL_GAGE, edgecolor="white",
    linewidth=1.0, zorder=10,
)

# Basemap
try:
    import contextily as ctx
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=10, alpha=0.45)
except Exception:
    ax.set_facecolor("#EAF2F8")

ax.set_axis_off()

# ── Inset ──────────────────────────────────────────────────────────────────────
basin.plot(ax=ax_ins, facecolor=COL_BASIN_FILL, edgecolor=COL_BASIN_EDGE, linewidth=0.8)

# Great Salt Lake
gsl = lakes[lakes["Lake_name"] == "Great Salt"].to_crs(TARGET_CRS)
if not gsl.empty:
    gsl.plot(ax=ax_ins, facecolor=COL_LAKE, edgecolor=COL_LAKE_EDGE,
             linewidth=0.5, alpha=0.9, zorder=2)
    ctr = gsl.geometry.iloc[0].centroid
    ax_ins.annotate("Great\nSalt Lake", xy=(ctr.x, ctr.y),
                    ha="center", va="center",
                    fontsize=5.5, color="#1A5276", fontweight="bold", zorder=6)

if not upstream_catch.empty:
    upstream_catch.plot(ax=ax_ins, facecolor=COL_UPSTREAM, edgecolor=COL_UP_EDGE,
                        linewidth=0.2, alpha=0.85, zorder=3)
if not terminal_catch.empty:
    terminal_catch.plot(ax=ax_ins, facecolor=COL_TERMINAL, edgecolor=COL_TERM_EDGE,
                        linewidth=0.8, alpha=0.95, zorder=4)

# Highlighted well in inset
ax_ins.scatter(
    hl_row.geometry.x, hl_row.geometry.y,
    s=60, marker="o",
    facecolor=COL_HL_WELL, edgecolor=COL_HL_EDGE,
    linewidth=0.8, zorder=6,
)

# Gage in inset
ax_ins.scatter(
    trow.geometry.x, trow.geometry.y,
    s=200, marker="*",
    facecolor=COL_GAGE, edgecolor="white",
    linewidth=0.8, zorder=7,
)

ax_ins.set_aspect("equal")
ax_ins.set_axis_off()
ax_ins.set_title("Location within GSLB", fontsize=9, fontweight="bold", pad=4)
for spine in ax_ins.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("#888888")
    spine.set_linewidth(0.8)

# ── Legend panel ───────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=COL_UPSTREAM, edgecolor=COL_UP_EDGE, linewidth=1.0,
                   alpha=0.75, label=f"Upstream catchments ({len(upstream_only_ids)})"),
    mpatches.Patch(facecolor=COL_TERMINAL, edgecolor=COL_TERM_EDGE, linewidth=1.5,
                   label="Terminal gage catchment"),
    mlines.Line2D([0], [0], color=COL_STREAM, linewidth=2.0,
                  label=f"Streams ({len(streams_clip)})"),
    mpatches.Patch(facecolor=COL_LAKE, edgecolor=COL_LAKE_EDGE,
                   label=f"Lakes ({len(lakes_clip)})"),
    mlines.Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=COL_WELL, markersize=10,
                  markeredgecolor="white", markeredgewidth=0.5, linewidth=0,
                  label=f"Wells ({len(wells_other)})"),
    mlines.Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=COL_HL_WELL, markersize=10,
                  markeredgecolor=COL_HL_EDGE, markeredgewidth=0.8, linewidth=0,
                  label=f"Highlighted well\n({HIGHLIGHT_WELL_ID})"),
    mlines.Line2D([0], [0], marker="*", color="w",
                  markerfacecolor=COL_GAGE, markersize=16,
                  markeredgecolor="white", markeredgewidth=0.8, linewidth=0,
                  label=f"Terminal gage ({GAGE_ID})"),
]

ax_leg.set_axis_off()
leg = ax_leg.legend(
    handles=legend_elements,
    loc="upper center",
    fontsize=9, framealpha=0.95, edgecolor="#aaaaaa",
    title="Map Elements", title_fontsize=10,
    frameon=True, borderpad=1.0, handlelength=2.0, handleheight=1.2,
)
ax_leg.add_artist(leg)

# ── Title ──────────────────────────────────────────────────────────────────────
fig.suptitle(
    f"Terminal Gage {GAGE_ID}  —  {gage_name}",
    fontsize=13, fontweight="bold", y=0.975,
)

out_path = OUT / f"{GAGE_ID}_highlight_well_{HIGHLIGHT_WELL_ID}.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")
