"""
Round-1 revision, item 3 — Within-well R^2, n, and clustered 95% CI per gage
(reviewer_1_response.md, Response 1: "don't state a direction of change until
these numbers exist").

Produces one table per retained catchment (Bear River, Weber River, Provo
River, Little Cottonwood; Spanish Fork shown but flagged excluded) with three
estimators side by side, so the CI-widening argument in the letter is visible
end to end rather than asserted:

  1. naive pooled OLS       - the published Method 1 estimator (Table 2):
                              every well-month treated as an independent row.
  2. pooled OLS, clustered by well  - same slope as (1); only the SE changes.
                              Reproduces the letter's illustrative example:
                              Bear River's 95% CI widens from [3.79, 7.23] to
                              roughly [-15.37, 26.38] once clustered.
  3. well fixed effects, clustered  - the within estimator: removes each
                              well's own baseline before fitting, then
                              clusters by well. This is "pooling done
                              correctly" (MEMO_coauthors.md, Decision 1) and
                              is reported both on the raw delta and on the
                              deseasonalised delta (the Section 5 headline
                              spec, see round1_deseasonalised_fe_full.py).

All three carry R^2 (pooled/uncentered for 1-2, within/centered for 3, see
docstring in round1_deseasonalised_fe_full.py) so the "pooled R^2 is
structurally low" argument (Response 1a) can be read directly off this table.

Run:  ./.venv/bin/python notebooks/round1_r2_n_ci_table.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE = Path(__file__).parent.parent
SRC = BASE / "results" / "features" / "data_with_deltas.csv"
FE_CSV = BASE / "results" / "round1_revision" / "01_deseasonalised_within_well" / "deseasonalised_fe_full.csv"
OUT = BASE / "results" / "round1_revision" / "03_r2_n_ci_table"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}
RETAINED = ["Bear River", "Weber River", "Provo River", "Little Cottonwood", "Spanish Fork"]


def pooled_fits(g):
    y = g["delta_q"].to_numpy(dtype=float)
    X = sm.add_constant(g["delta_wte"].to_numpy(dtype=float))
    well = g["well_id"].astype(str).values

    naive = sm.OLS(y, X).fit()
    n = len(g)
    tcrit_naive = stats.t.ppf(0.975, n - 2)
    naive_ci = (naive.params[1] - tcrit_naive * naive.bse[1],
                naive.params[1] + tcrit_naive * naive.bse[1])

    clu = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": well})
    n_clusters = len(set(well))
    tcrit_clu = stats.t.ppf(0.975, max(n_clusters - 1, 1))
    clu_ci = (clu.params[1] - tcrit_clu * clu.bse[1],
              clu.params[1] + tcrit_clu * clu.bse[1])

    return {
        "naive_slope": naive.params[1], "naive_r2": naive.rsquared,
        "naive_p": naive.pvalues[1], "naive_ci_lo": naive_ci[0], "naive_ci_hi": naive_ci[1],
        "naive_n": n,
        "clustered_slope": clu.params[1], "clustered_p": clu.pvalues[1],
        "clustered_ci_lo": clu_ci[0], "clustered_ci_hi": clu_ci[1],
        "n_wells": n_clusters,
    }


def main():
    d = pd.read_csv(SRC, parse_dates=["date"], low_memory=False)
    d["gage_id"] = d.gage_id.astype(str).str.replace(".0", "", regex=False)
    d["well_id"] = d.well_id.astype(str)
    d = d.dropna(subset=["delta_wte", "delta_q"])
    d = d[d.gage_id.isin(SHORT)].copy()

    rows = []
    for gid, g in d.groupby("gage_id"):
        gname = SHORT[gid]
        fit = pooled_fits(g)
        rows.append({"gage": gname, **fit})
    pooled = pd.DataFrame(rows).set_index("gage").loc[RETAINED].reset_index()

    fe = pd.read_csv(FE_CSV)
    fe0 = fe[fe.lag_months == 0]
    fe_raw = fe0[fe0.series == "raw"].set_index("gage")
    fe_ds = fe0[fe0.series == "deseasonalised"].set_index("gage")

    pd.set_option("display.width", 220)
    print("=" * 110)
    print("R1 COMMENT 1 — POOLED vs CLUSTERED vs WITHIN-WELL, PER GAGE")
    print("=" * 110)
    for _, row in pooled.iterrows():
        gname = row.gage
        flag = "" if gname != "Spanish Fork" else "  [EXCLUDED FROM BASIN SUM, <10 wells]"
        print(f"\n--- {gname} (n={int(row.naive_n):,} well-months, {int(row.n_wells)} wells){flag} ---")
        print(f"  naive pooled OLS         slope {row.naive_slope:+8.4f}  R^2={row.naive_r2:.4f}  "
              f"p={row.naive_p:.3g}  95% CI [{row.naive_ci_lo:+8.3f}, {row.naive_ci_hi:+8.3f}]")
        print(f"  pooled, clustered by well slope {row.clustered_slope:+8.4f}  R^2={row.naive_r2:.4f}  "
              f"p={row.clustered_p:.3g}  95% CI [{row.clustered_ci_lo:+8.3f}, {row.clustered_ci_hi:+8.3f}]"
              f"   <- SE-only change; slope identical to naive")
        if gname in fe_raw.index:
            fr = fe_raw.loc[gname]
            print(f"  well FE, clustered (raw)  slope {fr.slope:+8.4f}  R^2={fr.r2:.4f}  "
                  f"p={fr.p:.3g}  95% CI [{fr.ci_lo:+8.3f}, {fr.ci_hi:+8.3f}]")
        if gname in fe_ds.index:
            fd = fe_ds.loc[gname]
            print(f"  well FE, clustered (deseasonalised, FEATURED) "
                  f"slope {fd.slope:+8.4f}  R^2={fd.r2:.4f}  "
                  f"p={fd.p:.3g}  95% CI [{fd.ci_lo:+8.3f}, {fd.ci_hi:+8.3f}]")

    out = pooled.merge(
        fe_raw[["slope", "se", "p", "ci_lo", "ci_hi", "r2"]].add_prefix("fe_raw_"),
        left_on="gage", right_index=True, how="left"
    ).merge(
        fe_ds[["slope", "se", "p", "ci_lo", "ci_hi", "r2"]].add_prefix("fe_deseasonalised_"),
        left_on="gage", right_index=True, how="left"
    )
    out.to_csv(OUT / "r1_comment1_full_table.csv", index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
