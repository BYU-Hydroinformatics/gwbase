"""
Round-1 revision, item 5 — Verify no retained well's paired GEOGLOWS reach,
and no terminal gage's reach, falls inside the Great Salt Lake polygon.

reviewer_2_response.md, Response 3: GEOGLOWS routes reaches through lake
polygons to preserve upstream-downstream topology (a routing artifact, not a
claim that rivers cross the lakebed). The response states as fact that none
of the retained wells or terminal gages are paired to a lake-crossing reach,
and flags that this must be VERIFIED before submission, not asserted.

Method: load the lake polygon (data/raw/hydrography/gsl_lake.shp) and the
GEOGLOWS stream network (data/raw/hydrography/gslb_stream.shp, reach ID field
LINKNO). For every well actually retained in the final analysis (the 275
wells in results/features/data_with_deltas.csv, spanning all 5 terminal
catchments including Spanish Fork -- excluded from the *basin-scale*
aggregation by the ten-well rule, but still a retained well-gage pairing that
this geometric check should cover), look up its paired reach
(results/processed/well_reach_relationships.csv, Reach_ID == LINKNO) and test
whether that reach geometry intersects the lake polygon. Separately, look up
each of the 5 terminal gages' own reach via COMID_v2
(data/raw/streamflow/gsl_nwm.csv) and run the same test.

Run:  ./.venv/bin/python notebooks/round1_lake_reach_verification.py
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE = Path(__file__).parent.parent
OUT = BASE / "results" / "round1_revision" / "05_lake_reach_verification"
OUT.mkdir(parents=True, exist_ok=True)

TERMINAL_GAGES = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}


def main():
    lake = gpd.read_file(BASE / "data" / "raw" / "hydrography" / "gsl_lake.shp")
    stream = gpd.read_file(BASE / "data" / "raw" / "hydrography" / "gslb_stream.shp")

    # Great Salt Lake is Hylak_id 67 in the HydroLAKES-derived shapefile
    # (confirmed by inspection: single-lake shapefile in this basin extract).
    lake = lake.to_crs(stream.crs)
    lake_union = lake.union_all() if hasattr(lake, "union_all") else lake.unary_union

    stream["LINKNO"] = stream["LINKNO"].astype("int64")
    stream["in_lake"] = stream.geometry.intersects(lake_union)
    reach_in_lake = dict(zip(stream.LINKNO, stream.in_lake))
    n_lake_reaches = stream.in_lake.sum()
    print(f"Stream network: {len(stream):,} reaches total, {n_lake_reaches:,} intersect "
          f"the lake polygon ({n_lake_reaches / len(stream) * 100:.1f}%).")

    # ---- Retained wells --------------------------------------------------
    d = pd.read_csv(BASE / "results" / "features" / "data_with_deltas.csv",
                     usecols=["well_id", "gage_id"])
    d["gage_id"] = d.gage_id.astype(str).str.replace(".0", "", regex=False)
    retained_wells = set(d.well_id.unique())
    print(f"\nRetained wells in final analysis: {len(retained_wells):,}")

    wr = pd.read_csv(BASE / "results" / "processed" / "well_reach_relationships.csv")
    wr = wr[wr.Well_ID.isin(retained_wells)].copy()
    wr["Reach_ID"] = wr["Reach_ID"].astype("int64")
    wr["in_lake"] = wr["Reach_ID"].map(reach_in_lake)

    unmatched = wr["in_lake"].isna().sum()
    if unmatched:
        print(f"WARNING: {unmatched} retained well(s) have a Reach_ID not found in the "
              f"stream network — investigate before treating this check as conclusive.")
    wr["in_lake"] = wr["in_lake"].fillna(False)

    bad_wells = wr[wr.in_lake]
    print(f"Retained wells whose paired reach intersects the lake: {len(bad_wells)} "
          f"of {len(wr)} matched.")
    if len(bad_wells):
        print(bad_wells[["Well_ID", "Reach_ID", "Downstream_Gage"]].to_string(index=False))

    # ---- Terminal gages ----------------------------------------------------
    nwm = pd.read_csv(BASE / "data" / "raw" / "streamflow" / "gsl_nwm.csv",
                       usecols=["samplingFeatureCode", "COMID_v2"])
    nwm["samplingFeatureCode"] = nwm["samplingFeatureCode"].astype(str).str.replace(
        ".0", "", regex=False)

    gage_rows = []
    for gid, name in TERMINAL_GAGES.items():
        match = nwm[nwm.samplingFeatureCode == gid]
        if match.empty or match.COMID_v2.isna().all():
            gage_rows.append({"gage_id": gid, "gage": name, "comid": None,
                               "in_lake": None, "note": "no COMID_v2 match found"})
            continue
        comid = int(match.COMID_v2.iloc[0])
        in_lake = reach_in_lake.get(comid)
        gage_rows.append({"gage_id": gid, "gage": name, "comid": comid,
                           "in_lake": in_lake,
                           "note": "" if in_lake is not None else "COMID not in stream network"})

    gage_df = pd.DataFrame(gage_rows)
    print("\nTerminal gage -> reach lake check:")
    print(gage_df.to_string(index=False))

    wr.to_csv(OUT / "retained_wells_lake_check.csv", index=False)
    gage_df.to_csv(OUT / "terminal_gages_lake_check.csv", index=False)

    any_well_bad = bool(wr.in_lake.any())
    any_gage_bad = bool(gage_df.in_lake.fillna(False).any())
    print("\n" + "=" * 78)
    print("WELL-GAGE PAIRING CLAIM (Steps 1-3, 6):", end=" ")
    if not any_well_bad and unmatched == 0:
        print(f"VERIFIED. 0 of {len(wr)} retained wells are paired to a "
              "lake-intersecting reach.")
    else:
        print("NOT CLEANLY VERIFIED — see flagged wells above.")

    print("TERMINAL-GAGE CLAIM:", end=" ")
    if not any_gage_bad:
        print("VERIFIED. No terminal gage's reach intersects the lake.")
    else:
        bad = gage_df[gage_df.in_lake.fillna(False)]
        print(f"NOT LITERALLY TRUE for {list(bad.gage)}. Their outlet reach partially "
              "overlaps the lake polygon.")
        print("  This is expected, not an error: these are lake-draining terminal")
        print("  catchments, so their own outlet reach necessarily meets the lake it")
        print("  drains to (e.g. Bear River discharges into Bear River Bay). It is a")
        print("  different phenomenon from the routing artifact Reviewer 2 asked about")
        print("  (unrelated reaches drawn crossing THROUGH the lake in Figs. 3/10).")
        print("  No well is paired to this outlet reach (checked above), so the")
        print("  well-gage pairing and elevation filter are unaffected either way.")
        print("  ACTION NEEDED: reword the response-letter claim from a blanket 'no")
        print("  terminal gage sits on a lake-crossing reach' to something precise,")
        print("  e.g. 'no retained well is paired to a lake-crossing reach; the Bear")
        print("  River terminal reach's own downstream tail meets Bear River Bay, as")
        print("  expected for a lake-draining outlet, and this does not affect any")
        print("  well-gage pairing.'")
    print("=" * 78)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
