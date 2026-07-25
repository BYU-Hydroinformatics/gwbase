"""
Theil-Sen slopes and 95% confidence intervals per gage (Reviewer 1 R1 #2, Reviewer 3 A3(d)).

Both reviewers ask for robust regression and honest uncertainty:

  R1: "bootstrap or analytic 95% CIs on all reported slopes; Theil-Sen slopes for the
       same pairs."
  A3(d): "Theil-Sen has a breakdown point of ~29% and is insensitive to the high-leverage
       points the reviewer identifies, so a material divergence between the two estimators
       is direct evidence of outlier control -- and agreement is direct evidence against it."

Note A3(d)'s logic is backwards as drafted: agreement between OLS and Theil-Sen is evidence
that outliers are NOT driving the fit (i.e. evidence of control), and divergence is evidence
that they ARE. That wording needs fixing whichever way the numbers land.

Reported per gage:
  OLS slope with analytic and bootstrap 95% CI
  Theil-Sen slope with its distribution-free 95% CI
  cluster-robust CI (by well), because the naive CI is too narrow -- see handoff section 4c

The existing senslope_summary_table.csv is Sen's slope on WTE *time trends* (ft/yr), a
different quantity. This is Theil-Sen on the dWTE-dQ relationship.

Run:  ./.venv/bin/python notebooks/theilsen_and_cis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE = Path(__file__).parent.parent
SRC = BASE / "result" / "features" / "data_with_deltas.csv"
OUT = BASE / "result" / "analysis" / "theilsen_cis"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}
N_BOOT = 2000
SEED = 20260725


def cluster_bootstrap(g, rng, n_boot):
    """Resample WELLS, not rows -- rows within a well are not independent."""
    wells = g.well_id.unique()
    by_well = {w: sub for w, sub in g.groupby("well_id")}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(wells, size=len(wells), replace=True)
        s = pd.concat([by_well[w] for w in pick], ignore_index=True)
        if s.delta_wte.std() == 0:
            continue
        out.append(stats.linregress(s.delta_wte, s.delta_q).slope)
    return np.array(out)


def main():
    d = pd.read_csv(SRC, low_memory=False)
    d["gage_id"] = d.gage_id.astype(str).str.replace(".0", "", regex=False)
    d = d.dropna(subset=["delta_wte", "delta_q"])
    d["well_id"] = d.well_id.astype(str)
    rng = np.random.default_rng(SEED)

    rows = []
    for gid, g in d.groupby("gage_id"):
        if gid not in SHORT:
            continue
        x = g.delta_wte.to_numpy(float)
        y = g.delta_q.to_numpy(float)

        ols = stats.linregress(x, y)
        tcrit = stats.t.ppf(0.975, len(x) - 2)
        ols_lo, ols_hi = ols.slope - tcrit * ols.stderr, ols.slope + tcrit * ols.stderr

        # cluster-robust CI (by well)
        f = sm.OLS(y, sm.add_constant(x)).fit(
            cov_type="cluster", cov_kwds={"groups": g.well_id.values})
        cl_se = f.bse[1]
        cl_lo, cl_hi = f.conf_int()[1]

        ts = stats.theilslopes(y, x, alpha=0.95)

        boot = cluster_bootstrap(g, rng, N_BOOT)
        b_lo, b_hi = (np.percentile(boot, [2.5, 97.5]) if len(boot)
                      else (np.nan, np.nan))

        rows.append({
            "gage": SHORT[gid], "n_obs": len(g), "n_wells": g.well_id.nunique(),
            "ols_slope": ols.slope, "ols_ci_lo": ols_lo, "ols_ci_hi": ols_hi,
            "cluster_se": cl_se, "cluster_ci_lo": cl_lo, "cluster_ci_hi": cl_hi,
            "boot_ci_lo": b_lo, "boot_ci_hi": b_hi,
            "theilsen_slope": ts[0], "theilsen_ci_lo": ts[2], "theilsen_ci_hi": ts[3],
        })

    r = pd.DataFrame(rows).sort_values("ols_slope", ascending=False)
    r.to_csv(OUT / "theilsen_and_cis_by_gage.csv", index=False)

    print("\n=== SLOPES WITH 95% CONFIDENCE INTERVALS (cfs/ft) ===\n")
    print(f"{'gage':<19}{'OLS':>9}{'naive 95% CI':>22}{'well-clustered 95% CI':>26}")
    for _, x in r.iterrows():
        print(f"{x.gage:<19}{x.ols_slope:>9.3f}"
              f"{f'[{x.ols_ci_lo:.3f}, {x.ols_ci_hi:.3f}]':>22}"
              f"{f'[{x.cluster_ci_lo:.2f}, {x.cluster_ci_hi:.2f}]':>26}")

    print(f"\n{'gage':<19}{'cluster-boot 95% CI':>24}{'Theil-Sen':>12}{'TS 95% CI':>24}")
    for _, x in r.iterrows():
        print(f"{x.gage:<19}{f'[{x.boot_ci_lo:.2f}, {x.boot_ci_hi:.2f}]':>24}"
              f"{x.theilsen_slope:>12.4f}"
              f"{f'[{x.theilsen_ci_lo:.4f}, {x.theilsen_ci_hi:.4f}]':>24}")

    print("\n=== OLS vs THEIL-SEN (A3(d)) ===")
    for _, x in r.iterrows():
        agree = "same sign" if np.sign(x.ols_slope) == np.sign(x.theilsen_slope) else "SIGN DIFFERS"
        ratio = x.ols_slope / x.theilsen_slope if x.theilsen_slope else np.nan
        print(f"  {x.gage:<19} OLS {x.ols_slope:>8.3f}  TS {x.theilsen_slope:>8.4f}  "
              f"ratio {ratio:>8.1f}x  {agree}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
