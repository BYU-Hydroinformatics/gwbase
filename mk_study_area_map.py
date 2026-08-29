"""
Great Salt Lake Basin Study Area Map
"""

from pathlib import Path
import io
import urllib.request
import zipfile
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
HYD  = BASE / "data" / "raw" / "hydrography"
OUT  = BASE / "results" / "analysis" / "maps" / "overview"
OUT.mkdir(parents=True, exist_ok=True)

# ── Natural Earth helpers ──────────────────────────────────────────────────────
NE_STATES_DIR = Path.home() / ".cache" / "ne_50m_admin_1_states_provinces"
NE_STATES_SHP = NE_STATES_DIR / "ne_50m_admin_1_states_provinces.shp"

NE_LINES_DIR = Path.home() / ".cache" / "ne_50m_admin_1_states_provinces_lines"
NE_LINES_SHP = NE_LINES_DIR / "ne_50m_admin_1_states_provinces_lines.shp"


def _download_ne(url, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url.split('/')[-1]} (one-time)...")
    with urllib.request.urlopen(url) as r:
        with zipfile.ZipFile(io.BytesIO(r.read())) as z:
            z.extractall(dest_dir)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading shapefiles...")
basin   = gpd.read_file(HYD / "gsl_basin.shp")
rivers  = gpd.read_file(HYD / "Rivers.shp")
lakes   = gpd.read_file(HYD / "gsl_lake.shp")

print("Loading Natural Earth US states...")
states = conus = None
try:
    if not NE_STATES_SHP.exists():
        _download_ne(
            "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip",
            NE_STATES_DIR,
        )
    if not NE_LINES_SHP.exists():
        _download_ne(
            "https://naciscdn.org/naturalearth/50m/cultural/"
            "ne_50m_admin_1_states_provinces_lines.zip",
            NE_LINES_DIR,
        )
    _all = gpd.read_file(NE_STATES_SHP)
    states = _all[_all["iso_a2"] == "US"]
    conus  = states[~states["postal"].isin(["AK", "HI"])]
    print(f"  CONUS states: {len(conus)}")
except Exception as e:
    print(f"  States unavailable: {e}")

# ── Reproject to Web Mercator ──────────────────────────────────────────────────
CRS = "EPSG:3857"
basin_wm  = basin.to_crs(CRS)
rivers_wm = rivers.to_crs(CRS)
lakes_wm  = lakes.to_crs(CRS)
states_wm = states.to_crs(CRS) if states is not None else None
try:
    borders_wm = gpd.read_file(NE_LINES_SHP).to_crs(CRS)
except Exception:
    borders_wm = None

# ── Map extent with asymmetric padding ────────────────────────────────────────
# Extra padding on the RIGHT and BOTTOM creates white space for legend / scale
# bar / inset so they never overlap the basin features.
minx, miny, maxx, maxy = basin_wm.total_bounds
Wb = maxx - minx   # basin width  in WM metres
Hb = maxy - miny   # basin height in WM metres

LEFT_F   = 0.06
RIGHT_F  = 0.26   # trimmed: just enough for elements + north arrow
BOTTOM_F = 0.08
TOP_F    = 0.06

x0 = minx - LEFT_F   * Wb
x1 = maxx + RIGHT_F  * Wb
y0 = miny - BOTTOM_F * Hb
y1 = maxy + TOP_F    * Hb
W  = x1 - x0
H  = y1 - y0

# ── Precomputed axes-fraction thresholds ──────────────────────────────────────
# (used to position legend / scale bar / inset in the "white" margins)
BASIN_RIGHT_F  = (LEFT_F + 1) / (LEFT_F + 1 + RIGHT_F)   # ≈ 0.706
BASIN_BOTTOM_F = BOTTOM_F / (BOTTOM_F + 1 + TOP_F)        # ≈ 0.185

RS = BASIN_RIGHT_F + 0.015   # left edge of right-strip elements  ≈ 0.721

# ── Major cities ───────────────────────────────────────────────────────────────
CITIES = {
    "Logan":          (-111.834, 41.735),
    "Ogden":          (-111.973, 41.223),
    "Salt Lake City": (-111.891, 40.761),
    "Provo":          (-111.658, 40.234),
}
cities_gdf = gpd.GeoDataFrame(
    {"name": list(CITIES.keys())},
    geometry=gpd.points_from_xy(
        [v[0] for v in CITIES.values()],
        [v[1] for v in CITIES.values()],
    ),
    crs="EPSG:4326",
).to_crs(CRS)

# ── Figure & axes ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 9))
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_axis_off()
# Solid map border (state boundaries are dashed; the frame must not be)
ax.add_patch(mpatches.FancyBboxPatch(
    (0, 0), 1, 1, boxstyle="square,pad=0",
    facecolor="none", edgecolor="black", linewidth=0.8,
    transform=ax.transAxes, zorder=20, clip_on=False,
))

# ── Background ────────────────────────────────────────────────────────────────
ax.set_facecolor("#F2F4F5")

# ── State boundaries — clipped so far-left fragments don't appear ─────────────
if states_wm is not None:
    from shapely.geometry import box as _box
    _clip = gpd.GeoDataFrame(
        geometry=[_box(minx, y0, x1, y1)], crs=CRS
    )
    _states_vis = gpd.clip(states_wm, _clip)
    _view = _box(x0, y0, x1, y1)      # full drawn extent, so lines reach the edge
    if borders_wm is not None:
        _bl = borders_wm.geometry.intersection(_view)
    else:
        _bl = gpd.GeoSeries([states_wm.geometry.boundary.unary_union],
                            crs=CRS).intersection(_view)
    gpd.GeoSeries(_bl[~_bl.is_empty].values, crs=CRS).plot(
        ax=ax, color="#444444", linewidth=1.0, linestyle="--", zorder=2,
    )
    # State names, set in the open ground of each state's visible portion.
    for _, _st in _states_vis.iterrows():
        _g = _st.geometry
        if _g.is_empty:
            continue
        _name = str(_st["name"]).upper()
        # Utah and Nevada are placed by hand; their largest ice-free area sits
        # where the label collides with the basin outline or the legend.
        _fixed = {"UTAH": (0.511, 0.260), "NEVADA": (0.078, 0.422)}
        if _name in _fixed:
            ax.annotate(_name, xy=_fixed[_name], xycoords="axes fraction",
                        ha="center", va="center", fontsize=11, color="#8E9BA6",
                        fontweight="bold", zorder=3)
            continue
        _outside = _g.difference(basin_wm.unary_union)
        if _outside.is_empty:
            continue
        _parts = list(getattr(_outside, "geoms", [_outside]))
        _big   = max(_parts, key=lambda q: q.area)
        _p     = _big.representative_point()
        ax.annotate(_name, xy=(_p.x, _p.y),
                    ha="center", va="center", fontsize=11, color="#8E9BA6",
                    fontweight="bold", zorder=3)

# ── GSLB boundary ─────────────────────────────────────────────────────────────
basin_wm.plot(ax=ax, facecolor="none", edgecolor="#CC0000", linewidth=2.2, zorder=3)

# ── Rivers ────────────────────────────────────────────────────────────────────
rivers_wm.plot(ax=ax, color="#2166AC", linewidth=0.8, alpha=0.9, zorder=4)

# ── Lakes ─────────────────────────────────────────────────────────────────────
lakes_wm.plot(ax=ax, facecolor="#AED6F1", edgecolor="#5DADE2",
              linewidth=0.4, alpha=0.9, zorder=5)

# ── Cities ────────────────────────────────────────────────────────────────────
cities_gdf.plot(ax=ax, color="#E67E22", markersize=50, marker="o",
                edgecolor="black", linewidth=0.8, zorder=7)

LABEL_DX = 18_000   # m in WM
LABEL_DY =  6_000
for _, row in cities_gdf.iterrows():
    ax.annotate(
        row["name"],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(row.geometry.x + LABEL_DX, row.geometry.y + LABEL_DY),
        fontsize=8, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor="black", linewidth=0.7, alpha=0.95),
        va="bottom", zorder=8,
    )

# ── North arrow — placed just inside the basin's upper-right area ─────────────
# North arrow at top-right inside the map, near the eastern basin boundary
na_xf     = BASIN_RIGHT_F + 0.04   # just inside right strip, near basin edge
na_yf     = 0.875
na_x      = x0 + na_xf * W
na_y0_    = y0 + na_yf * H
na_len    = H * 0.038
circle_r  = H * 0.018
circle_cy = na_y0_ + na_len + circle_r * 1.10

ax.annotate("", xy=(na_x, na_y0_ + na_len), xytext=(na_x, na_y0_),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                            mutation_scale=14), zorder=9)
ax.add_patch(mpatches.Circle(
    (na_x, circle_cy), circle_r,
    facecolor="white", edgecolor="black", linewidth=1.2,
    transform=ax.transData, zorder=10,
))
ax.text(na_x, circle_cy, "N",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="black", zorder=11)

# ══════════════════════════════════════════════════════════════════════════════
# All cartographic elements below use ax.transAxes so they sit in the
# right (RS → 0.985) / bottom (0 → BASIN_BOTTOM_F) white-space strips.
# ══════════════════════════════════════════════════════════════════════════════
trans = ax.transAxes

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor="none", edgecolor="#CC0000", linewidth=2,
                   label="GSLB Boundary"),
    mlines.Line2D([0], [0], color="#444444", linewidth=1.0, linestyle="--",
                  label="State Boundary"),
    mlines.Line2D([0], [0], color="#2166AC", linewidth=1.2,
                  label="Major Rivers"),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E67E22",
                  markersize=7, markeredgecolor="white", label="Major Cities"),
    mpatches.Patch(facecolor="#AED6F1", edgecolor="#5DADE2", linewidth=0.5,
                   label="Lakes"),
]
# ── Stack layout (axes fraction, bottom → top) ────────────────────────────────
# Raise the block so it sits comfortably in the right strip,
# not crammed at the bottom.
#
#   INS_Y0 ──┐ inset  (height INS_H; inset title adds ~0.015 above)
#             └ top ≈ INS_Y0+INS_H+0.015
#   GAP 0.018
#   _sb_bot ──┐ scale-bar backing
#   BAR_Y      │ black bar (height 0.010)
#   _sb_top ──┘ (bar + label clearance)
#   GAP 0.022
#   LEG_Y ────  legend lower-right anchor

# ── Bottom-margin stack (all axes fraction, bottom → top) ────────────────────
# BASIN_BOTTOM_F ≈ 0.28/(0.28+1+0.06) ≈ 0.207  — elements must stay below this
# to appear "under the basin".  Stack from bottom of figure upward:
#   inset  →  gap  →  scale bar  →  gap  →  legend
# Elements left-edge at the basin's eastern boundary; stack in lower-right quadrant
INS_Y0  = 0.015
INS_H   = 0.125      # enlarged locator inset
_sb_bot = 0.165          # pinned: legend and scale bar must not move
BAR_Y   = _sb_bot + 0.038   # moved up closer to legend
_sb_top = BAR_Y   + 0.010 + 0.018
LEG_Y   = _sb_top + 0.006

ins_x0 = BASIN_RIGHT_F - 0.075   # shift well left, overlapping with basin right area
ins_w  = 0.234                    # keep same width
RIGHT_EDGE = ins_x0 + ins_w

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(
    handles=legend_handles,
    loc="lower center",
    fontsize=8,
    title="Legend",
    title_fontsize=8.5,
    framealpha=0.92,
    edgecolor="#aaaaaa",
    frameon=True,
    borderpad=0.7,
    labelspacing=0.35,
    bbox_to_anchor=(ins_x0 + ins_w / 2, LEG_Y),  # centre-aligned with scale bar & inset
)

# ── Measure rendered legend width → use it for scale bar & inset ─────────────
fig.canvas.draw()
_leg  = ax.get_legend()
_lb   = _leg.get_window_extent()
_ab   = ax.get_window_extent()
el_x0 = (_lb.x0 - _ab.x0) / _ab.width   # legend left  in axes fraction
el_w  = _lb.width            / _ab.width  # legend width in axes fraction

# ── Scale bar (100 km) ────────────────────────────────────────────────────────
scale_m  = 130_500            # 100 km corrected for WM at ~40°N
bar_w_f  = scale_m / W
bar_x0_f = el_x0 + (el_w - bar_w_f) / 2   # centre bar within the frame

# Black bar only — no backing box, single "100 km" label centred below
ax.add_patch(mpatches.FancyBboxPatch(
    (bar_x0_f, BAR_Y), bar_w_f, 0.010,
    boxstyle="square,pad=0", transform=trans,
    facecolor="#333333", edgecolor="#333333", zorder=10,
))
ax.text(bar_x0_f + bar_w_f / 2, BAR_Y - 0.006, "100 km",
        ha="center", va="top", fontsize=7, transform=trans, zorder=10, color="#333333")

# ── USA location inset — pixel-exact same width as legend via fig.add_axes ────
# Convert legend display-px bounds → figure fraction so width matches exactly.
_fb   = fig.get_window_extent()
INS_FRAC        = 0.34        # share of the map frame width
_ins_fig_width  = INS_FRAC * _ab.width / _fb.width
_ins_fig_left   = (_ab.x1 - _fb.x0) / _fb.width - _ins_fig_width - 0.012
_cb = conus.to_crs("EPSG:5070").total_bounds if conus is not None else (0, 0, 1.55, 1.0)
_ins_fig_height = (_ins_fig_width * _fb.width
                   * ((_cb[3] - _cb[1]) / (_cb[2] - _cb[0])) / _fb.height)
_ins_fig_bottom = (_ab.y0 - _fb.y0) / _fb.height + 0.012
ax_ins = fig.add_axes([_ins_fig_left, _ins_fig_bottom,
                        _ins_fig_width, _ins_fig_height])
ax_ins.set_facecolor("none")

try:
    if conus is None:
        raise RuntimeError("CONUS states not available")

    conus_alb = conus.to_crs("EPSG:5070")
    basin_alb = basin.to_crs("EPSG:5070")

    conus_alb.plot(ax=ax_ins, facecolor="#D5D8DC", edgecolor="white", linewidth=0.4)

    # Red filled rectangle = GSLB bounding box (visible at CONUS scale)
    bx0_b, by0_b, bx1_b, by1_b = basin_alb.total_bounds
    ax_ins.add_patch(mpatches.Rectangle(
        (bx0_b, by0_b), bx1_b - bx0_b, by1_b - by0_b,
        facecolor="#CC0000", edgecolor="#CC0000", linewidth=0.5, alpha=0.95, zorder=5,
    ))

    bx0, by0, bx1, by1 = conus_alb.total_bounds
    px, py = (bx1 - bx0) * 0.02, (by1 - by0) * 0.02
    ax_ins.set_xlim(bx0 - px, bx1 + px)
    ax_ins.set_ylim(by0 - py, by1 + py)
    # No set_aspect — let CONUS fill the full allocated width
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    for sp in ax_ins.spines.values():
        sp.set_visible(False)

except Exception as e:
    print(f"  Inset skipped: {e}")
    ax_ins.text(0.5, 0.5, "Location in USA", ha="center", va="center",
                fontsize=7, transform=ax_ins.transAxes)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUT / "gslb_study_area.png"
plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path}")
plt.show()
