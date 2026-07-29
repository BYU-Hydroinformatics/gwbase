"""
Round-1 revision, item 4 — A6 baseflow-magnitude comparison, updated.

Reviewer 3, A6: the 3.4-9.8 cfs/ft sensitivity is "within the error of Q and
inconsequential" against peak flows of several thousand cfs. The response
pushes back on the denominator (baseflow, not peak flow, since the estimate
is derived from baseflow-dominated months) and reframes against actual
observed multi-foot WTE declines rather than a nominal 1 ft.

This is the same computation as notebooks/a6_baseflow_context.py, updated
for two things settled since that script last ran:
  1. Spanish Fork is dropped from the basin sum (ten-well threshold).
  2. The sensitivity value being contextualized is now the deseasonalised
     within-well basin sum (round1_basin_sum_four_catchments.py), not the
     old Method-1-clipped-to-Method-5 range.

NOTE: this script does not decide the argument either. It computes the
numbers; if the new basin-sum point estimate is a negligible fraction of
baseflow, or its own CI crosses zero, that must be stated plainly rather
than talked around.

Run:  ./.venv/bin/python notebooks/round1_a6_baseflow_context.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
import gwbase  # noqa: E402

RAW = BASE / "data" / "raw"
BASIN_SUM_CSV = BASE / "results" / "round1_revision" / "02_basin_sum_four_catchments" / "basin_sum_summary.csv"
OUT = BASE / "results" / "round1_revision" / "04_a6_baseflow_context"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10163000": "Provo River", "10168000": "Little Cottonwood",
}  # Spanish Fork dropped, ten-well threshold


def main():
    basin = pd.read_csv(BASIN_SUM_CSV).iloc[0]
    sens_point = basin.basin_slope_cfs_per_ft
    sens_lo, sens_hi = basin.ci_lo, basin.ci_hi
    old_lo, old_hi = basin.old_range_lo, basin.old_range_hi

    sf = gwbase.load_streamflow_data(
        str(RAW / "streamflow" / "gages_with_bfd_predictions"), filter_bfd=False)
    sf["gage_id"] = sf["gage_id"].astype(str).str.replace(".0", "", regex=False)
    sf = sf[sf.gage_id.isin(SHORT)].dropna(subset=["q"]).copy()
    sf["date"] = pd.to_datetime(sf["date"])

    rows = []
    for gid, g in sf.groupby("gage_id"):
        b = g[g.bfd == 1]
        rows.append({
            "gage": SHORT[gid],
            "bfd_mean_q": b.q.mean(),
            "bfd_median_q": b.q.median(),
            "all_mean_q": g.q.mean(),
            "peak_p99": g.q.quantile(0.99),
            "peak_max": g.q.max(),
        })
    t = pd.DataFrame(rows).sort_values("bfd_mean_q", ascending=False)

    basin_bfd_mean = t.bfd_mean_q.sum()
    basin_bfd_median = t.bfd_median_q.sum()
    basin_all_mean = t.all_mean_q.sum()
    basin_peak = t.peak_p99.sum()

    pd.set_option("display.width", 200)
    print("\n=== BFD-MONTH DISCHARGE BY GAGE (cfs) — 4 retained catchments ===")
    print(t.round(1).to_string(index=False))

    print("\n=== BASIN SUMS (cfs) — 4 retained catchments ===")
    print(f"  baseflow (sum of per-gage mean BFD-day q)   : {basin_bfd_mean:>10,.1f}")
    print(f"  baseflow (sum of per-gage median BFD-day q) : {basin_bfd_median:>10,.1f}")
    print(f"  all-day mean flow                           : {basin_all_mean:>10,.1f}")
    print(f"  99th-percentile flow (Reviewer 3's scale)   : {basin_peak:>10,.1f}")

    print("\n=== NEW SENSITIVITY AS A FRACTION OF EACH DENOMINATOR ===")
    print(f"  Deseasonalised within-well basin sum: point {sens_point:+.3f} cfs/ft, "
          f"95% CI [{sens_lo:+.3f}, {sens_hi:+.3f}]")
    print(f"  Old published range: {old_lo}-{old_hi} cfs/ft")
    print(f"\n{'denominator':<44}{'point est. % per ft':>22}{'CI-based % range':>28}")
    for lbl, den in (("basin baseflow (mean BFD q)", basin_bfd_mean),
                     ("basin baseflow (median BFD q)", basin_bfd_median),
                     ("basin all-day mean flow", basin_all_mean),
                     ("basin 99th-pct flow (peak framing)", basin_peak)):
        pt_pct = sens_point / den * 100
        lo_pct, hi_pct = sens_lo / den * 100, sens_hi / den * 100
        print(f"  {lbl:<42}{pt_pct:>18.2f}%   [{lo_pct:>7.2f}%, {hi_pct:>7.2f}%]")

    # ---- implied reduction at OBSERVED decline rates -------------------------
    mk = BASE / "results" / "features" / "mk_well_wte.csv"
    print("\n=== IMPLIED REDUCTION AT OBSERVED DECLINE RATES (point-estimate sensitivity) ===")
    decline_rows = []
    if mk.is_file():
        w = pd.read_csv(mk)
        scol = next((c for c in w.columns if "slope" in c.lower()), None)
        if scol:
            med = w[scol].median()
            dec = w[w[scol] < 0][scol]
            print(f"  median Sen's slope, all wells      : {med:+.4f} ft/yr")
            print(f"  median among declining wells       : {dec.median():+.4f} ft/yr "
                  f"({len(dec):,} of {len(w):,} wells declining)")
            for yrs in (10, 20, 30):
                for lbl, rate in (("all wells", med), ("declining only", dec.median())):
                    drop = abs(rate) * yrs
                    reduction = sens_point * drop
                    pct = reduction / basin_bfd_mean * 100
                    lo_r, hi_r = sens_lo * drop, sens_hi * drop
                    print(f"    {yrs:>2}yr @ {lbl:<15} decline {drop:>5.2f} ft -> "
                          f"{reduction:>6.2f} cfs point-est ({pct:5.2f}% of baseflow); "
                          f"CI-based [{lo_r:>6.2f}, {hi_r:>6.2f}] cfs")
                    decline_rows.append({"years": yrs, "rate_basis": lbl, "decline_ft": drop,
                                          "reduction_cfs_point": reduction,
                                          "reduction_pct_of_baseflow": pct,
                                          "reduction_cfs_ci_lo": lo_r, "reduction_cfs_ci_hi": hi_r})
    else:
        print(f"  (mk_well_wte.csv not found at {mk})")

    # ---- 2011-2016 drought context (manuscript line 132, ref35: ~10 km3 loss) ----
    print("\n=== 2011-2016 DROUGHT STORAGE-LOSS CONTEXT (manuscript ref35: ~10 km^3) ===")
    print("  This figure is a basin-wide GPS-deformation storage-volume estimate, not a")
    print("  WTE decline, and no basin specific-yield value is available in this project")
    print("  to convert km^3 <-> feet of water-table decline. It is retained in the")
    print("  manuscript as background motivation (Section 2) rather than converted into")
    print("  a cfs comparison here; the Sen's-slope-based decline rates above are the")
    print("  well-observation-based equivalent of 'an actual observed decline' and are")
    print("  the basis for the argument, not the km^3 figure.")

    t.to_csv(OUT / "bfd_discharge_by_gage.csv", index=False)
    pd.DataFrame([{
        "basin_baseflow_mean_cfs": basin_bfd_mean,
        "basin_baseflow_median_cfs": basin_bfd_median,
        "basin_all_mean_cfs": basin_all_mean,
        "basin_p99_cfs": basin_peak,
        "sens_point_cfs_per_ft": sens_point,
        "sens_ci_lo": sens_lo,
        "sens_ci_hi": sens_hi,
        "sens_point_pct_of_baseflow": sens_point / basin_bfd_mean * 100,
        "sens_ci_lo_pct_of_baseflow": sens_lo / basin_bfd_mean * 100,
        "sens_ci_hi_pct_of_baseflow": sens_hi / basin_bfd_mean * 100,
    }]).to_csv(OUT / "a6_summary.csv", index=False)
    if decline_rows:
        pd.DataFrame(decline_rows).to_csv(OUT / "implied_reduction_by_decline_rate.csv", index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
