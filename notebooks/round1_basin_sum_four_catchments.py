"""
Round-1 revision, item 2 — Basin sum over the four retained catchments,
under the deseasonalised within-well specification.

Replaces the published 3.4-9.8 cfs/ft range (Method 1 after 99th-pct clipping,
to Method 5 rolling-12m — see gwbase_manuscript_10.tex lines 413/469/490/503),
which is dropped for two independent reasons argued in the response letters:
  - Method 5's p=0.0 at all five gages is a serial-correlation artifact
    (rolling-mean construction), not signal (see notebooks/rolling12m_delta_
    analysis.py and reviewer_1_response.md, Response 2).
  - Spanish Fork (7 wells) is now excluded from ALL basin-scale aggregation
    by the new ten-well catchment threshold (reviewer_1_response.md, Response 3).

Basin sum = sum of per-gage deseasonalised, well-fixed-effects, clustered
slopes (cfs/ft) over the four retained catchments (Bear River, Weber River,
Provo River, Little Cottonwood), at zero lag -- the same specification
featured in Section 5.

The point estimate is a single number, not a range, because there is now one
primary specification rather than a spread across six methods. A defensible
*range* is instead constructed from the basin sum's own 95% CI, propagated
from the four per-gage clustered SEs assuming cross-catchment independence
(each gage's discharge series and well set is catchment-specific, so this is
a reasonable assumption, unlike the pseudo-replication problem within a
catchment that motivated clustering in the first place).

Depends on: notebooks/round1_deseasonalised_fe_full.py having been run first
(reads its output CSV).

Run:  ./.venv/bin/python notebooks/round1_basin_sum_four_catchments.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).parent.parent
SRC = BASE / "results" / "round1_revision" / "01_deseasonalised_within_well" / "deseasonalised_fe_full.csv"
OUT = BASE / "results" / "round1_revision" / "02_basin_sum_four_catchments"
OUT.mkdir(parents=True, exist_ok=True)

RETAINED = ["Bear River", "Weber River", "Provo River", "Little Cottonwood"]


def main():
    r = pd.read_csv(SRC)
    z = r[(r.lag_months == 0) & (r.series == "deseasonalised") & (r.gage.isin(RETAINED))].copy()
    z = z.set_index("gage").loc[RETAINED].reset_index()

    basin_slope = z.slope.sum()
    # Propagate SE assuming independence across catchments (distinct gages,
    # distinct well sets, distinct discharge series).
    basin_se = np.sqrt((z.se ** 2).sum())
    # Conservative dof: the smallest per-gage cluster count minus 1.
    dof = int(z.n_wells.min() - 1)
    tcrit = stats.t.ppf(0.975, dof)
    ci_lo, ci_hi = basin_slope - tcrit * basin_se, basin_slope + tcrit * basin_se

    print("=" * 78)
    print("PER-GAGE CONTRIBUTIONS (deseasonalised within-well, zero lag)")
    print("=" * 78)
    for _, row in z.iterrows():
        pct = row.slope / basin_slope * 100 if basin_slope else np.nan
        print(f"  {row.gage:<19} {row.slope:+7.3f} cfs/ft  "
              f"(se {row.se:.3f}; {pct:5.1f}% of basin sum; {int(row.n_wells)} wells)")

    print("\n" + "=" * 78)
    print("BASIN SUM — 4 retained catchments, deseasonalised within-well spec")
    print("=" * 78)
    print(f"  Point estimate      : {basin_slope:+.3f} cfs/ft")
    print(f"  95% CI (propagated) : [{ci_lo:+.3f}, {ci_hi:+.3f}] cfs/ft  (dof={dof})")
    print(f"  Old published range : 3.4 - 9.8 cfs/ft (Method 1 clipped -> Method 5, "
          f"5 gages incl. Spanish Fork)")

    old_lo, old_hi = 3.4, 9.8
    print(f"\n  New point estimate is "
          f"{'within' if old_lo <= basin_slope <= old_hi else 'outside'} "
          f"the old published range.")

    pd.DataFrame([{
        "n_catchments": len(RETAINED),
        "basin_slope_cfs_per_ft": basin_slope,
        "basin_se": basin_se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "dof": dof,
        "old_range_lo": old_lo,
        "old_range_hi": old_hi,
    }]).to_csv(OUT / "basin_sum_summary.csv", index=False)
    z.to_csv(OUT / "per_gage_contributions.csv", index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
