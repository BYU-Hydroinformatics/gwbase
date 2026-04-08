"""
For each terminal gage, produce one map showing:
  • The gage's own catchment (highlighted in orange)
  • All upstream catchments (light green)
  • Streams and wells within those catchments
  • Wells colored by Mutual Information (white = no MI data; viridis colormap = has MI data)
  • Top-10 MI wells highlighted with callout annotation boxes
  • Right-panel: inset overview (top) + colorbar & legend (bottom)
Output: results/figures/terminal_gage_maps_mi/
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
DATA    = BASE / "data"
RESULTS = BASE / "results"
OUT     = RESULTS / "figures" / "terminal_gage_maps_mi"
OUT.mkdir(parents=True, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────────
COL_UPSTREAM   = "#C7E9C0"
COL_UP_EDGE    = "#74C476"
COL_TERMINAL   = "#FDAE6B"
COL_TERM_EDGE  = "#D94801"
COL_STREAM     = "#2171B5"
COL_LAKE       = "#9ECAE1"
COL_LAKE_EDGE  = "#6BAED6"
COL_WELL_GREY  = "white"       # wells with no MI data
COL_GAGE       = "#E31A1C"
COL_BASIN_FILL = "#EEEEEE"
COL_BASIN_EDGE = "#888888"
COL_TOP_EDGE   = "#FFCC00"    # yellow ring for top-10 wells

WELL_CMAP = cm.viridis         # colormap for MI
TOP_N     = 10                 # number of top wells to annotate

# ── Load geodata ──────────────────────────────────────────────────────────────
print("Loading geodata...")
basin     = gpd.read_file(DATA / "raw/hydrography/gsl_basin.shp")
catchment = gpd.read_file(DATA / "raw/hydrography/gsl_catchment.shp")
streams   = gpd.read_file(DATA / "raw/hydrography/gslb_stream.shp")
lakes     = gpd.read_file(DATA / "raw/hydrography/lake.shp")
wells_gdf = gpd.read_file(DATA / "raw/hydrography/well_shp.shp")

all_gages = pd.read_csv(DATA    / "raw/hydrography/gsl_nwm_gage.csv")
term_df   = pd.read_csv(RESULTS / "processed/terminal_gages.csv")
upstream  = pd.read_csv(RESULTS / "processed/terminal_gage_upstream_catchments.csv")

# MI per well
mi_df = pd.read_csv(RESULTS / "features/mi_analysis.csv")
mi_df["well_id_str"] = mi_df["well_id"].astype(str)

# Fixed colorbar range 0–1.5 (covers global MI max ~1.46)
MI_CBAR_MIN = 0.0
MI_CBAR_MAX = 1.5
norm = mcolors.Normalize(vmin=MI_CBAR_MIN, vmax=MI_CBAR_MAX)

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
mi_global_min = mi_df["mi"].min()
mi_global_max = mi_df["mi"].max()
print(f"MI data: {len(mi_df)} wells across {mi_df['gage_id'].nunique()} gages "
      f"(range {mi_global_min:.3f}–{mi_global_max:.3f})")

# ── Great Salt Lake (for inset) ───────────────────────────────────────────────
gsl = lakes[lakes["Lake_name"] == "Great Salt"].to_crs(TARGET_CRS)
gsl_centroid = gsl.geometry.iloc[0].centroid if not gsl.empty else None


def annotation_offsets(n):
    base = [
        ( 70,  35), (-70,  35), ( 70, -35), (-70, -35),
        (100,   0), (-100,  0), (  0,  70), (  0, -70),
        ( 85,  60), (-85,  60), ( 85, -60), (-85, -60),
    ]
    while len(base) < n:
        base += [(dx * 1.5, dy * 1.5) for dx, dy in base]
    return base[:n]


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

    # ── Attach MI values to wells ─────────────────────────────────────────────
    wells_clip = wells_clip.copy()
    wells_clip["well_id_str"] = (
        wells_clip["Well_ID"].astype(float).astype(int).astype(str)
    )
    gage_mi = mi_df[mi_df["gage_id"].astype(str) == str(gage_id)].copy()

    wells_clip = wells_clip.merge(
        gage_mi[["well_id_str", "mi"]],
        on="well_id_str", how="left",
    )
    has_mi = wells_clip[wells_clip["mi"].notna()].copy()
    no_mi  = wells_clip[wells_clip["mi"].isna()].copy()

    top_wells = has_mi.nlargest(min(TOP_N, len(has_mi)), "mi") if not has_mi.empty else pd.DataFrame()
    print(f"  Wells total: {len(wells_clip)}  |  with MI: {len(has_mi)}  |  top {TOP_N}: {len(top_wells)}")

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        width_ratios=[0.72, 0.28],
        height_ratios=[0.45, 0.30, 0.25],
        hspace=0.05, wspace=0.02,
        left=0.01, right=0.99, top=0.92, bottom=0.04,
    )
    ax      = fig.add_subplot(gs[:, 0])
    ax_ins  = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[1, 1])
    ax_leg  = fig.add_subplot(gs[2, 1])

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

    if not no_mi.empty:
        xs = [g.x for g in no_mi.geometry]
        ys = [g.y for g in no_mi.geometry]
        ax.scatter(xs, ys, s=28, c=COL_WELL_GREY, marker="o",
                   edgecolors="#AAAAAA", linewidths=0.5, alpha=0.85, zorder=7)

    if not has_mi.empty:
        xs     = [g.x for g in has_mi.geometry]
        ys     = [g.y for g in has_mi.geometry]
        colors = [WELL_CMAP(norm(v)) for v in has_mi["mi"]]
        ax.scatter(xs, ys, s=28, c=colors, marker="o",
                   edgecolors="white", linewidths=0.4, alpha=0.85, zorder=8)

    if not top_wells.empty:
        top_xs     = [g.x for g in top_wells.geometry]
        top_ys     = [g.y for g in top_wells.geometry]
        top_colors = [WELL_CMAP(norm(v)) for v in top_wells["mi"]]
        ax.scatter(top_xs, top_ys, s=90, c=top_colors, marker="o",
                   edgecolors=COL_TOP_EDGE, linewidths=1.5, zorder=9)

        offsets = annotation_offsets(len(top_wells))
        for i, (_, wrow) in enumerate(top_wells.iterrows()):
            dx, dy = offsets[i]
            ax.annotate(
                f"MI={wrow['mi']:.3f}",
                xy=(wrow.geometry.x, wrow.geometry.y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.5, fontweight="bold",
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=COL_TOP_EDGE, alpha=0.92, linewidth=1.0),
                arrowprops=dict(arrowstyle="-|>", color=COL_TOP_EDGE,
                                lw=0.8, mutation_scale=8),
                zorder=11,
            )

    ax.scatter(trow.geometry.x, trow.geometry.y, s=600, marker="*",
               facecolor=COL_GAGE, edgecolor="white", linewidth=1.0, zorder=12)

    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=10, alpha=0.45)
    except Exception:
        ax.set_facecolor("#EAF2F8")
    ax.set_axis_off()

    # ── Inset ─────────────────────────────────────────────────────────────────
    basin.plot(ax=ax_ins, facecolor=COL_BASIN_FILL, edgecolor=COL_BASIN_EDGE,
               linewidth=0.8)
    if not gsl.empty:
        gsl.plot(ax=ax_ins, facecolor=COL_LAKE, edgecolor=COL_LAKE_EDGE,
                 linewidth=0.5, alpha=0.9, zorder=2)
        if gsl_centroid:
            ax_ins.annotate("Great\nSalt Lake", xy=(gsl_centroid.x, gsl_centroid.y),
                            ha="center", va="center", fontsize=5.5,
                            color="#1A5276", fontweight="bold", zorder=6)
    if not upstream_catch.empty:
        upstream_catch.plot(ax=ax_ins, facecolor=COL_UPSTREAM, edgecolor=COL_UP_EDGE,
                            linewidth=0.2, alpha=0.85, zorder=3)
    if not terminal_catch.empty:
        terminal_catch.plot(ax=ax_ins, facecolor=COL_TERMINAL, edgecolor=COL_TERM_EDGE,
                            linewidth=0.8, alpha=0.95, zorder=4)
    ax_ins.scatter(trow.geometry.x, trow.geometry.y, s=200, marker="*",
                   facecolor=COL_GAGE, edgecolor="white", linewidth=0.8, zorder=7)
    ax_ins.set_aspect("equal")
    ax_ins.set_axis_off()
    ax_ins.set_title("Location within GSLB", fontsize=9, fontweight="bold", pad=4)
    for spine in ax_ins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.8)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    ax_cbar.set_axis_off()
    if not has_mi.empty:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax_cbar, width="85%", height="35%", loc="center")
        cb  = ColorbarBase(cax, cmap=WELL_CMAP, norm=norm,
                           orientation="horizontal")
        cb.set_label("Mutual Information (ΔWTE vs ΔQ)", fontsize=8)
        ticks = np.linspace(MI_CBAR_MIN, MI_CBAR_MAX, 6)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.1f}" for t in ticks])
        cb.ax.tick_params(labelsize=7)
        gage_mi_min = has_mi["mi"].min()
        gage_mi_max = has_mi["mi"].max()
        cax.text(0.0, -0.7, f"map min={gage_mi_min:.3f}", transform=cax.transAxes,
                 fontsize=6.5, color="#555555", ha="left", va="top")
        cax.text(1.0, -0.7, f"map max={gage_mi_max:.3f}", transform=cax.transAxes,
                 fontsize=6.5, color="#555555", ha="right", va="top")
        ax_cbar.set_title(
            f"Wells with MI data: {len(has_mi)}  |  Top {len(top_wells)} highlighted",
            fontsize=7.5, pad=2,
        )
    else:
        ax_cbar.text(0.5, 0.5, "No MI data\nfor this gage",
                     ha="center", va="center", fontsize=9,
                     color="#888888", transform=ax_cbar.transAxes)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=COL_UPSTREAM, edgecolor=COL_UP_EDGE,
                       linewidth=1.0, alpha=0.75,
                       label=f"Upstream catchments ({len(upstream_only_ids)})"),
        mpatches.Patch(facecolor=COL_TERMINAL, edgecolor=COL_TERM_EDGE,
                       linewidth=1.5, label="Terminal gage catchment"),
        mlines.Line2D([0], [0], color=COL_STREAM, linewidth=2.0,
                      label=f"Streams ({len(streams_clip)})"),
        mpatches.Patch(facecolor=COL_LAKE, edgecolor=COL_LAKE_EDGE,
                       label=f"Lakes ({len(lakes_clip)})"),
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=COL_WELL_GREY, markersize=8,
                      markeredgecolor="#AAAAAA", markeredgewidth=0.5, linewidth=0,
                      label=f"Wells – no MI ({len(no_mi)})"),
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=WELL_CMAP(0.75), markersize=8,
                      markeredgecolor="white", linewidth=0,
                      label=f"Wells – with MI ({len(has_mi)})"),
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=WELL_CMAP(0.95), markersize=10,
                      markeredgecolor=COL_TOP_EDGE, markeredgewidth=1.5,
                      linewidth=0,
                      label=f"Top {len(top_wells)} MI wells"),
        mlines.Line2D([0], [0], marker="*", color="w",
                      markerfacecolor=COL_GAGE, markersize=14,
                      markeredgecolor="white", linewidth=0,
                      label=f"Terminal gage ({gage_id})"),
    ]
    ax_leg.set_axis_off()
    leg = ax_leg.legend(handles=legend_elements, loc="upper center",
                        fontsize=8, framealpha=0.95, edgecolor="#aaaaaa",
                        title="Map Elements", title_fontsize=9,
                        frameon=True, borderpad=0.8,
                        handlelength=1.8, handleheight=1.1)
    ax_leg.add_artist(leg)

    fig.suptitle(f"Terminal Gage {gage_id}  —  {gage_name}",
                 fontsize=13, fontweight="bold", y=0.975)

    out_path = OUT / f"{gage_id}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")

print(f"\nDone. Maps saved to: {OUT}")
