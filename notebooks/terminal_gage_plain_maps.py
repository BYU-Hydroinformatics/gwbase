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
    # The legend is built and measured first. Its rendered width sets the width
    # of the whole right-hand column, which in turn sets how much room the main
    # map gets, so nothing is sized by guesswork. The locator inset then takes
    # that same width and whatever height the basin's own proportions require,
    # so its frame fits its contents rather than floating in a fixed cell.
    FIG_W, FIG_H = 15, 11
    RIGHT_MARGIN = 0.015     # figure fraction kept clear to the right
    PANEL_GAP    = 0.030     # vertical space between inset and legend
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=COL_UPSTREAM, edgecolor=COL_UP_EDGE,
                       linewidth=1.0, alpha=0.75,
                       label=f"Upstream catchments ({len(upstream_only_ids)})"),
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
    ax_leg = fig.add_axes([0.75, 0.30, 0.22, 0.30])
    ax_leg.set_axis_off()
    leg = ax_leg.legend(handles=legend_elements,
                        loc="upper left", bbox_to_anchor=(0, 1),
                        fontsize=12, framealpha=0.95, edgecolor="#aaaaaa",
                        title="Map Elements", title_fontsize=14,
                        frameon=True, borderpad=0.9, borderaxespad=0.0,
                        labelspacing=0.75, handlelength=2.2, handleheight=1.4)
    ax_leg.add_artist(leg)

    # ── Size the right-hand column from the rendered legend ───────────────────
    fig.canvas.draw()
    _fb   = fig.get_window_extent()
    _lb   = leg.get_window_extent()
    box_w = _lb.width  / _fb.width
    leg_h = _lb.height / _fb.height
    box_l = 1.0 - RIGHT_MARGIN - box_w

    # Inset height that makes its frame fit the basin exactly at equal scale.
    _bx0, _by0, _bx1, _by1 = basin.total_bounds
    ins_h = box_w * (FIG_W / FIG_H) * ((_by1 - _by0) / (_bx1 - _bx0))

    _stack_h = ins_h + PANEL_GAP + leg_h
    _stack_b = (1.0 - _stack_h) / 2.0
    ax_leg.set_position([box_l, _stack_b, box_w, leg_h])
    ax_ins = fig.add_axes([box_l, _stack_b + leg_h + PANEL_GAP, box_w, ins_h])

    # Main map takes everything to the left of the column.
    ax = fig.add_axes([0.005, 0.02, box_l - 0.020, 0.96])

    # ── Main map ──────────────────────────────────────────────────────────────
    if not lakes_clip.empty:
        lakes_clip.plot(ax=ax, facecolor=COL_LAKE, edgecolor=COL_LAKE_EDGE,
                        linewidth=0.4, alpha=0.75, zorder=1)
    if not upstream_catch.empty:
        upstream_catch.plot(ax=ax, facecolor=COL_UPSTREAM, edgecolor=COL_UP_EDGE,
                            linewidth=0.3, alpha=0.75, zorder=2)
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
    ax.scatter(trow.geometry.x, trow.geometry.y, s=260, marker="*",
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
        from shapely.geometry import box as _box, Point
        from shapely.ops import polylabel
        view_box = _box(ax.get_xlim()[0], ax.get_ylim()[0],
                         ax.get_xlim()[1], ax.get_ylim()[1])
        gsl_geom = gsl.geometry.iloc[0]
        if gsl_geom.intersects(view_box):
            gsl_visible = gsl_geom.intersection(view_box)
            if not gsl_visible.is_empty:
                # Label the largest visible lobe of the lake, at its pole of
                # inaccessibility (the interior point furthest from any shore),
                # and set the name on two lines. representative_point() can land
                # in a narrow arm, which pushed the one-line label out over land.
                parts = list(getattr(gsl_visible, "geoms", [gsl_visible]))
                lobe  = max(parts, key=lambda g: g.area)
                # Chosen anchor in the southern main body of the lake, in the
                # plotting CRS. Every geometric rule tried here (centroid,
                # representative point, pole of inaccessibility) lands in the
                # northern arm instead, because that arm really is the widest
                # open water; it is not where the label reads best. Falls back
                # to a computed point if the anchor is outside the visible area.
                GSL_LABEL_XY = (-12535651, 5029082)
                label_pt = Point(*GSL_LABEL_XY)
                if not lobe.contains(label_pt):
                    try:
                        label_pt = polylabel(lobe, tolerance=100)
                    except Exception:
                        label_pt = lobe.representative_point()
                ax.annotate(
                    "Great\nSalt Lake", xy=(label_pt.x, label_pt.y),
                    ha="center", va="center", fontsize=15, fontweight="bold",
                    color="#1A5276", zorder=6, linespacing=1.15,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="none", alpha=0.6),
                )
    ax.set_axis_off()
    ax.set_anchor("E")

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
    # The axes box was already given the basin's own proportions, so an equal
    # aspect leaves the frame exactly where it was placed.
    ax_ins.set_aspect("equal", adjustable="datalim")
    ax_ins.set_xlim(_bx0, _bx1)
    ax_ins.set_ylim(_by0, _by1)
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

    out_path = OUT / f"{gage_id}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")

print(f"\nDone. Maps saved to: {OUT}")
