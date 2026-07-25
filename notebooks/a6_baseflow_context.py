"""
A6: is 3.4-9.8 cfs/ft "inconsequential"? Put it in the right denominator.

Reviewer 3: "-3.4 - -9.8 cfs/foot drawdown, which is both within the error of Q and
inconsequential in management decisions where absolute peak flows over several thousand
CFS are the norm."

The paper's claim (manuscript line 490) is about BASEFLOW, not total or peak flow:
"A 1 ft decline in water-table elevation averaged across the basin would thus reduce
total baseflow by roughly 3.4-9.8 cfs."

So the honest comparison is against basin baseflow during BFD months, not against
snowmelt peaks. This script computes that denominator and expresses the sensitivity
against it, alongside the peak-flow denominator Reviewer 3 used, so both are visible.

It also computes the implied reduction for an OBSERVED rate of water-table decline
rather than a hypothetical 1 ft, since that is what a manager would actually face.

NOTE — this script does not decide the argument. If the sensitivity really is a
negligible fraction of baseflow, that must be reported and the management framing
softened. See the A6 note in reviewer_3_response.md.

Run:  ./.venv/bin/python notebooks/a6_baseflow_context.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
import gwbase  # noqa: E402

RAW = BASE / "data" / "raw"
OUT = BASE / "result" / "analysis" / "a6_baseflow_context"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}
SENS_LOW, SENS_HIGH = 3.4, 9.8          # published basin-sum range, cfs/ft


def main():
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
    print("\n=== BFD-MONTH DISCHARGE BY GAGE (cfs) ===")
    print(t.round(1).to_string(index=False))

    print("\n=== BASIN SUMS (cfs) ===")
    print(f"  baseflow (sum of per-gage mean BFD-day q)   : {basin_bfd_mean:>10,.1f}")
    print(f"  baseflow (sum of per-gage median BFD-day q) : {basin_bfd_median:>10,.1f}")
    print(f"  all-day mean flow                           : {basin_all_mean:>10,.1f}")
    print(f"  99th-percentile flow (Reviewer 3's scale)   : {basin_peak:>10,.1f}")

    print("\n=== 3.4-9.8 cfs/ft AS A FRACTION OF EACH DENOMINATOR ===")
    print(f"{'denominator':<44}{'per 1 ft':>22}")
    for lbl, den in (("basin baseflow (mean BFD q)", basin_bfd_mean),
                     ("basin baseflow (median BFD q)", basin_bfd_median),
                     ("basin all-day mean flow", basin_all_mean),
                     ("basin 99th-pct flow (peak framing)", basin_peak)):
        lo, hi = SENS_LOW / den * 100, SENS_HIGH / den * 100
        print(f"  {lbl:<42}{lo:>8.2f}% - {hi:.2f}%")

    # ---- implied reduction at an OBSERVED decline rate -----------------------
    mk = BASE / "result" / "features" / "mk_well_wte.csv"
    print("\n=== IMPLIED REDUCTION AT OBSERVED DECLINE RATES ===")
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
                    lo, hi = SENS_LOW * drop, SENS_HIGH * drop
                    pct_lo, pct_hi = lo / basin_bfd_mean * 100, hi / basin_bfd_mean * 100
                    print(f"    {yrs:>2}yr @ {lbl:<15} decline {drop:>5.2f} ft -> "
                          f"{lo:>5.2f}-{hi:>5.2f} cfs "
                          f"({pct_lo:.2f}%-{pct_hi:.2f}% of baseflow)")
    else:
        print(f"  (mk_well_wte.csv not found at {mk} — run step 9 first)")

    t.to_csv(OUT / "bfd_discharge_by_gage.csv", index=False)
    pd.DataFrame([{
        "basin_baseflow_mean_cfs": basin_bfd_mean,
        "basin_baseflow_median_cfs": basin_bfd_median,
        "basin_all_mean_cfs": basin_all_mean,
        "basin_p99_cfs": basin_peak,
        "sens_low_pct_of_baseflow": SENS_LOW / basin_bfd_mean * 100,
        "sens_high_pct_of_baseflow": SENS_HIGH / basin_bfd_mean * 100,
    }]).to_csv(OUT / "a6_summary.csv", index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
