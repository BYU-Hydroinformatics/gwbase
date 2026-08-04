"""
Round-1 revision -- Section 6.2 trend counts, Spanish Fork excluded.

notebooks/round1_hamed_rao_trend_test.py reports 216/273 significant (plain),
161/273 significant (Hamed-Rao), 116 declining (plain), 83 declining (Hamed-Rao)
-- but that "273" still includes Spanish Fork's 7 wells, which contradicts the
manuscript's claim that Spanish Fork was dropped. This reruns the same two
tests (same source series, results/features/data_with_deltas.csv) restricted
to the four retained catchments (Bear, Weber, Provo, Little Cottonwood).

Also redone: the "153 of 273" declining-wells / -0.267 ft/yr median figure
quoted in a response letter (results/round1_revision/04_a6_baseflow_context,
sourced from results/features/mk_well_wte.csv). That count is over ALL wells
regardless of trend significance, not just the significant ones -- it needed
the same Spanish Fork exclusion, done separately from the plain/Hamed-Rao
significant-trend counts above.

Run:  ./.venv/bin/python notebooks/round1_trend_counts_no_spanish_fork.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk

BASE = Path(__file__).parent.parent
OUT = BASE / "results" / "round1_revision" / "12_trend_counts_no_spanish_fork"
OUT.mkdir(parents=True, exist_ok=True)

MIN_OBS = 10
SPANISH_FORK = 10152000


def run_both_tests(x):
    r_plain = mk.original_test(x)
    r_hr = mk.hamed_rao_modification_test(x)
    return r_plain, r_hr


def main():
    df = pd.read_csv(BASE / "results" / "features" / "data_with_deltas.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["gage_id"] != SPANISH_FORK].copy()
    df = df.sort_values(["well_id", "date"])

    rows = []
    for wid, grp in df.groupby("well_id"):
        grp = grp.dropna(subset=["wte"]).sort_values("date")
        if len(grp) < MIN_OBS:
            continue
        r_plain, r_hr = run_both_tests(grp["wte"].values)

        # ft/step -> ft/yr, same conversion as gwbase.analysis.compute_mk_well_wte
        # (needed to reproduce the "-0.267 ft/yr" style magnitude, not just the
        # significant/declining counts, which don't depend on the slope's scale).
        intervals = grp["date"].diff().dt.days.dropna()
        median_interval_days = intervals.median() if len(intervals) else np.nan
        if median_interval_days and median_interval_days > 0:
            slope_plain_yr = r_plain.slope * (365.25 / median_interval_days)
        else:
            slope_plain_yr = np.nan

        rows.append({
            "well_id": wid, "gage_id": grp["gage_id"].iloc[0], "n_obs": len(grp),
            "p_plain": r_plain.p, "slope_plain": r_plain.slope,
            "slope_plain_yr": slope_plain_yr, "trend_plain": r_plain.trend,
            "p_hamed_rao": r_hr.p, "slope_hamed_rao": r_hr.slope, "trend_hamed_rao": r_hr.trend,
        })

    res = pd.DataFrame(rows)
    n = len(res)

    sig_plain = res["p_plain"] < 0.05
    dec_plain = sig_plain & (res["slope_plain"] < 0)
    sig_hr = res["p_hamed_rao"] < 0.05
    dec_hr = sig_hr & (res["slope_hamed_rao"] < 0)

    print(f"Wells tested: {n} (four retained catchments, Spanish Fork excluded)")
    print(f"  Plain Mann-Kendall:  significant={sig_plain.sum()} of {n} "
          f"({100*sig_plain.sum()/n:.1f}%)  declining={dec_plain.sum()}")
    print(f"  Hamed-Rao:           significant={sig_hr.sum()} of {n} "
          f"({100*sig_hr.sum()/n:.1f}%)  declining={dec_hr.sum()}")
    print()
    print("For comparison, all-catchment (Spanish Fork included) §6.2 numbers were:")
    print("  216/273 significant plain, 161/273 significant Hamed-Rao, "
          "116 declining plain, 83 declining Hamed-Rao.")

    res.to_csv(OUT / "mk_well_wte_plain_vs_hamed_rao_no_spanish_fork.csv", index=False)
    print(f"\nWrote {OUT / 'mk_well_wte_plain_vs_hamed_rao_no_spanish_fork.csv'}")

    # ---- redo the "153 of 273" declining-wells / median-rate figure ----------
    # This is the plain-test Sen's-slope declining count over ALL wells
    # (regardless of significance), as quoted in the A6 baseflow-context
    # response letter -- not the same quantity as dec_plain above.
    print()
    print("=== Redo of the '153 of 273 declining, median -0.267 ft/yr' figure ===")
    all_dec = res[res["slope_plain_yr"] < 0]["slope_plain_yr"]
    print(f"  declining wells (plain Sen's slope < 0): {len(all_dec)} of {n}")
    print(f"  median decline rate among declining wells: {all_dec.median():+.4f} ft/yr")
    print(f"  (median Sen's slope, all wells): {res['slope_plain_yr'].median():+.4f} ft/yr")

    pd.DataFrame([{
        "n_wells": n,
        "n_declining_plain_sen_slope": len(all_dec),
        "median_decline_rate_ft_per_yr": all_dec.median(),
        "median_sen_slope_all_wells_ft_per_yr": res["slope_plain_yr"].median(),
    }]).to_csv(OUT / "declining_wells_median_rate_no_spanish_fork.csv", index=False)
    print(f"Wrote {OUT / 'declining_wells_median_rate_no_spanish_fork.csv'}")


if __name__ == "__main__":
    main()
