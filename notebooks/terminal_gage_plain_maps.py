"""
For each terminal gage, produce one plain map (no R² coloring) showing:
  • The gage's own catchment (highlighted in orange)
  • All upstream catchments (light green)
  • Streams, lakes, and wells within those catchments
  • The terminal gage point (large star)
  • Right-panel inset showing watershed location within the GSLB
  • Legend placed in right panel below inset
Output: results/figures/terminal_gage_maps/
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
DATA    = BASE / "data"
RESULTS = BASE / "results"
OUT     = RESULTS / "analysis" / "maps" / "plain"
OUT.mkdir(parents=True, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────────
COL_UPSTREAM   = "#C7E9C0"
COL_UP_EDGE    = "#74C476"
COL_TERMINAL   = "#FDAE6B"
COL_TERM_EDGE  = "#D94801"
COL_STREAM     = "#2171B5"
COL_LAKE       = "#9ECAE1"
COL_LAKE_EDGE  = "#6BAED6"
COL_WELL       = "#8B4513"
COL_GAGE       = "#E31A1C"
COL_BASIN_FILL = "#EEEEEE"
COL_BASIN_EDGE = "#888888"

# ── Load geodata ──────────────────────────────────────────────────────────────
print("Loading geodata...")
basin     = gpd.read_file(DATA / "raw/hydrography/gsl_basin.shp")
catchment = gpd.read_file(DATA / "raw/hydrography/gsl_catchment.shp")
streams   = gpd.read_file(DATA / "raw/hydrography/gslb_stream.shp")
lakes     = gpd.read_file(DATA / "raw/hydrography/lake.shp")
wells_gdf = gpd.read_file(DATA / "raw/hydrography/well_shp.shp")

all_gages = pd.read_csv(DATA    / "raw/hydrography/gsl_nwm_gage.csv")
term_df   = pd.read_csv(RESULTS / "processed" / "terminal_gages.csv")
upstream  = pd.read_csv(RESULTS / "processed" / "terminal_gage_upstream_catchments.csv")

TARGET_CRS = "EPSG:3857"
basin     = basin.to_crs(TARGET_CRS)
catchment = catchment.to_crs(TARGET_CRS)
streams   = streams.to_crs(TARGET_CRS)
lakes     = lakes.to_crs(TARGET_CRS)
wells_gdf = wells_gdf.to_crs(TARGET_CRS)

catchment["linkno_int"] = catchment["linkno"].astype(float).astype(int)

term_info = term_df.merge(
    all_gages[["id", "latitude", "longitude", "name"]],
    on="id", how="left", suffixes=("", "_gage"),
)
term_info["display_name"] = term_info["name"].fillna(
    term_info.get("name_gage", term_info["name"])
)
term_gdf = gpd.GeoDataFrame(
    term_info,
    geometry=gpd.points_from_xy(term_info["longitude"], term_info["latitude"]),
    crs="EPSG:4326",
).to_crs(TARGET_CRS)

print(f"Terminal gages: {len(term_gdf)}")

gsl = lakes[lakes["Lake_name"] == "Great Salt"].to_crs(TARGET_CRS)
gsl_centroid = gsl.geometry.iloc[0].centroid if not gsl.empty else None

# ── Per-gage plotting ─────────────────────────────────────────────────────────
for _, trow in term_gdf.iterrows():
    gage_id   = trow["id"]
    gage_name = trow["display_name"]
    print(f"\nPlotting: {gage_id} – {gage_name}")

    terminal_catch_id = int(float(trow["catchment_id"]))
    up_ids = (
        upstream.loc[upstream["Gage_ID"] == gage_id, "Upstream_Catchment_ID"]
        .astype(float).astype(int).tolist()
    )
    all_catch_ids     = set(up_ids) | {terminal_catch_id}
    upstream_only_ids = set(up_ids) - {terminal_catch_id}

    terminal_catch = catchment[catchment["linkno_int"] == terminal_catch_id].copy()
    upstream_catch = catchment[catchment["linkno_int"].isin(upstream_only_ids)].copy()
    all_catch      = catchment[catchment["linkno_int"].isin(all_catch_ids)].copy()

    if all_catch.empty:
        print(f"  WARNING: No catchment polygons found – skipping.")
        continue

    total_area   = all_catch.dissolve()
    total_geom   = total_area.geometry.iloc[0]
    streams_clip = streams[streams.intersects(total_geom)].copy()
    wells_clip   = wells_gdf[wells_gdf.within(total_geom)].copy()
    lakes_clip   = lakes[lakes.intersects(total_geom)].copy()

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[0.68, 0.32],
        height_ratios=[0.42, 0.58],
        hspace=0.05, wspace=0.02,
        left=0.01, right=0.99, top=0.99, bottom=0.04,
    )
    ax     = fig.add_subplot(gs[:, 0])
    ax_ins = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[1, 1])

    # ── Main map ──────────────────────────────────────────────────────────────
    if not lakes_clip.empty:
        lakes_clip.plot(ax=ax, facecolor=COL_LAKE, edgecolor=COL_LAKE_EDGE,
                        linewidth=0.4, alpha=0.75, zorder=1)
    if not upstream_catch.empty:
        upstream_catch.plot(ax=ax, facecolor=COL_UPSTREAM, edgecolor=COL_UP_EDGE,
                            linewidth=0.3, alpha=0.75, zorder=2)
    if not terminal_catch.empty:
        terminal_catch.plot(ax=ax, facecolor=COL_TERMINAL, edgecolor=COL_TERM_EDGE,
                            linewidth=1.5, alpha=0.90, zorder=3)
    total_area.plot(ax=ax, facecolor="none", edgecolor="#1A3A5C",
                    linewidth=1.2, zorder=4)
    if not streams_clip.empty:
        streams_clip.plot(ax=ax, color=COL_STREAM, linewidth=0.7,
                          alpha=0.85, zorder=5)
    if not wells_clip.empty:
        xs = [g.x for g in wells_clip.geometry]
        ys = [g.y for g in wells_clip.geometry]
        ax.scatter(xs, ys, s=28, c=COL_WELL, marker="o",
                   edgecolors="white", linewidths=0.4, alpha=0.75, zorder=7)
    ax.scatter(trow.geometry.x, trow.geometry.y, s=600, marker="*",
               facecolor=COL_GAGE, edgecolor="white", linewidth=1.0, zorder=10)

    try:
        import contextily as ctx
        # CartoDB's free anonymous tile tier now returns "API KEY REQUIRED"
        # watermarks for uncached tiles; Esri's gray canvas is label-free,
        # visually equivalent, and works without a key.
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas,
                        zoom=10, alpha=0.55)
    except Exception:
        ax.set_facecolor("#EAF2F8")

    # ── Great Salt Lake label — only drawn on the main map, and only when the
    #    lake actually falls inside this gage's plotted extent (the inset
    #    copy is too small to be legible, so it lives here instead).
    if not gsl.empty:
        from shapely.geometry import box as _box
        view_box = _box(ax.get_xlim()[0], ax.get_ylim()[0],
                         ax.get_xlim()[1], ax.get_ylim()[1])
        gsl_geom = gsl.geometry.iloc[0]
        if gsl_geom.intersects(view_box):
            gsl_visible = gsl_geom.intersection(view_box)
            if not gsl_visible.is_empty:
                label_pt = gsl_visible.representative_point()
                ax.annotate(
                    "Great Salt Lake", xy=(label_pt.x, label_pt.y),
                    ha="center", va="center", fontsize=15, fontweight="bold",
                    color="#1A5276", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="none", alpha=0.65),
                )
    ax.set_axis_off()

    # ── Inset ─────────────────────────────────────────────────────────────────
    basin.plot(ax=ax_ins, facecolor=COL_BASIN_FILL, edgecolor=COL_BASIN_EDGE,
               linewidth=0.8)
    if not gsl.empty:
        gsl.plot(ax=ax_ins, facecolor=COL_LAKE, edgecolor=COL_LAKE_EDGE,
                 linewidth=0.5, alpha=0.9, zorder=2)
    if not upstream_catch.empty:
        upstream_catch.plot(ax=ax_ins, facecolor=COL_UPSTREAM, edgecolor=COL_UP_EDGE,
                            linewidth=0.2, alpha=0.85, zorder=3)
    if not terminal_catch.empty:
        terminal_catch.plot(ax=ax_ins, facecolor=COL_TERMINAL, edgecolor=COL_TERM_EDGE,
                            linewidth=0.8, alpha=0.95, zorder=4)
    ax_ins.scatter(trow.geometry.x, trow.geometry.y, s=200, marker="*",
                   facecolor=COL_GAGE, edgecolor="white", linewidth=0.8, zorder=7)
    ax_ins.set_aspect("equal", anchor="W")   # hug the main map instead of centering
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    for spine in ax_ins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.2)

    # Black rectangle marking the watershed extent in the inset
    minx, miny, maxx, maxy = total_area.total_bounds
    ax_ins.add_patch(mpatches.Rectangle(
        (minx, miny), maxx - minx, maxy - miny,
        facecolor="none", edgecolor="#CC0000", linewidth=1.0, zorder=8,
    ))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=COL_UPSTREAM, edgecolor=COL_UP_EDGE,
                       linewidth=1.0, alpha=0.75,
                       label=f"Upstream catchments ({len(upstream_only_ids)})"),
        mpatches.Patch(facecolor=COL_TERMINAL, edgecolor=COL_TERM_EDGE,
                       linewidth=1.5, label="Terminal gage catchment"),
        mlines.Line2D([0], [0], color=COL_STREAM, linewidth=2.5,
                      label=f"Streams ({len(streams_clip)})"),
        mpatches.Patch(facecolor=COL_LAKE, edgecolor=COL_LAKE_EDGE,
                       label=f"Lakes ({len(lakes_clip)})"),
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=COL_WELL, markersize=14,
                      markeredgecolor="white", markeredgewidth=0.5,
                      linewidth=0, label=f"Wells ({len(wells_clip)})"),
        mlines.Line2D([0], [0], marker="*", color="w",
                      markerfacecolor=COL_GAGE, markersize=22,
                      markeredgecolor="white", markeredgewidth=0.8,
                      linewidth=0, label=f"Terminal gage ({gage_id})"),
    ]
    ax_leg.set_axis_off()
    leg = ax_leg.legend(handles=legend_elements, loc="center left",
                        bbox_to_anchor=(0.0, 0.5),
                        fontsize=15, framealpha=0.95, edgecolor="#aaaaaa",
                        title="Map Elements", title_fontsize=17,
                        frameon=True, borderpad=1.4,
                        labelspacing=1.1, handlelength=2.8, handleheight=1.8)
    ax_leg.add_artist(leg)


    out_path = OUT / f"{gage_id}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")

print(f"\nDone. Maps saved to: {OUT}")
