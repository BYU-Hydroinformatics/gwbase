"""
A4: how many continuous groundwater recorders sit inside the terminal-gage catchments?

Reviewer 3: "The temporal analyses, in my opinion, are at the wrong scale. In highly
connected systems, stream and groundwater responses are on the order of days to weeks,
not months... I feel this is completely lost by interpolating to monthly data."

The response promises to check whether a daily sub-analysis is even feasible. 65 sites
basin-wide have NWIS daily values (parameter 72019). The question is how many fall inside
the five terminal-gage contributing areas — which needs the catchment polygons, delivered
with Xueyi's data.

The letter is explicit that a negative answer is a legitimate finding:
"If the qualifying sample proves too small to support the analysis, say so explicitly in
the response — that is a legitimate finding and is far better than a vague claim."

Run:  ./.venv/bin/python notebooks/a4_recorders_in_catchments.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

BASE = Path(__file__).parent.parent
PAPER = Path.home() / "papers_git" / "2026_xueyi_GSLB_paper"
RDB = PAPER / "review1" / "retrieved_data" / "gslb_continuous_gw_recorders.rdb"
HYD = BASE / "data" / "raw" / "hydrography"
PROC = BASE / "result" / "processed"
OUT = BASE / "result" / "analysis" / "a4_recorders"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}


def read_rdb(path):
    """USGS RDB: '#' comments, header row, then a type row, then data."""
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    hdr = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[2:] if ln.strip()]
    return pd.DataFrame(rows, columns=hdr)


def main():
    sites = read_rdb(RDB)
    for c in ("dec_lat_va", "dec_long_va"):
        sites[c] = pd.to_numeric(sites[c], errors="coerce")
    sites = sites.dropna(subset=["dec_lat_va", "dec_long_va"])
    print(f"Continuous GW recorders pulled basin-wide: {len(sites)}")

    gdf = gpd.GeoDataFrame(
        sites,
        geometry=[Point(x, y) for x, y in zip(sites.dec_long_va, sites.dec_lat_va)],
        crs="EPSG:4326")

    catch = gpd.read_file(HYD / "gsl_catchment.shp")
    print(f"Catchment polygons: {len(catch):,}  (crs {catch.crs})")
    if catch.crs is not None and catch.crs.to_epsg() != 4326:
        catch = catch.to_crs(4326)

    # Which catchments drain to each terminal gage (Step 1 output).
    up = pd.read_csv(PROC / "terminal_gage_upstream_catchments.csv")
    print(f"Upstream-catchment table: {len(up):,} rows, columns {list(up.columns)[:6]}")

    # Must join on the UPSTREAM catchments (the full contributing area), not the
    # single terminal catchment -- otherwise only the outlet polygon is searched.
    gcol = "Gage_ID"
    ccol = "Upstream_Catchment_ID"
    up[gcol] = up[gcol].astype(str).str.replace(".0", "", regex=False)
    print(f"  using gage column '{gcol}', catchment column '{ccol}'")

    key = next((c for c in catch.columns
                if c.lower() in (ccol.lower(), "comid", "linkno", "streamlink")), None)
    if key is None:
        print(f"  !! no matching key in catchment shapefile: {list(catch.columns)}")
        return
    print(f"  joining on catchment key '{key}'")

    joined = gpd.sjoin(gdf, catch[[key, "geometry"]], how="inner", predicate="within")
    print(f"\nRecorders falling inside ANY basin catchment: {len(joined)}")

    # linkno is float64 in the shapefile and the catchment ids are int64 in the CSV,
    # so a plain astype(str) yields "710117152.0" vs "710117152" and matches nothing.
    def as_int_str(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    joined[key] = as_int_str(joined[key])
    up[ccol] = as_int_str(up[ccol])
    m = joined.merge(up[[gcol, ccol]], left_on=key, right_on=ccol, how="inner")
    m = m[m[gcol].isin(SHORT)]

    print("\n=== RECORDERS WITHIN TERMINAL-GAGE CATCHMENTS ===")
    if len(m) == 0:
        print("  NONE at any of the five terminal gages.")
    else:
        for gid, g in m.groupby(gcol):
            print(f"  {SHORT.get(gid, gid):<20} {g.site_no.nunique():>3} recorder(s)")
    print(f"\n  TOTAL across the five terminal catchments: {m.site_no.nunique()}")

    m.drop(columns="geometry").to_csv(OUT / "recorders_in_terminal_catchments.csv",
                                      index=False)
    print(f"\nWrote {OUT}")
    print("\nA small or zero count is a legitimate, reportable finding — see the A4 note.")


if __name__ == "__main__":
    main()
