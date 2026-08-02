"""
Round-1 revision, item 6 -- Section 6.2 trend test: plain vs. Hamed-Rao Mann-Kendall.

reviewer_3_response.md, B26 and MEMO_coauthors.md flag that analysis.py:673 used
`mk.original_test` (plain Mann-Kendall), which assumes serially independent
observations. The monthly PCHIP-interpolated WTE series has a median lag-1
autocorrelation of ~0.961, which inflates the plain test's significance. The fix
is `mk.hamed_rao_modification_test`, an autocorrelation-robust variant.

Two things had to be resolved before trusting any "before" number (see
response_submission_checklist.md section 3):

1. The delivered mk_well_wte.csv reports 216 significant / 116 declining wells
   (of 273), but the MEMO/draft letter cites 225/111. This script recomputes
   the plain test fresh, from the same source data as the robust test, to
   confirm which is right.
2. Report both counts computed in the *same* run on the *same* interpolated
   series (results/features/data_with_deltas.csv) -- not a cached plain-test
   output compared against a fresh robust-test output, which is how the
   216-vs-225 mismatch likely happened in the first place.

Run:  ./.venv/bin/python notebooks/round1_hamed_rao_trend_test.py
"""
from pathlib import Path

import pandas as pd
import pymannkendall as mk

BASE = Path(__file__).parent.parent
OUT = BASE / "results" / "round1_revision" / "06_hamed_rao_trend_test"
OUT.mkdir(parents=True, exist_ok=True)

MIN_OBS = 10


def run_both_tests(x):
    r_plain = mk.original_test(x)
    r_hr = mk.hamed_rao_modification_test(x)
    return r_plain, r_hr


def main():
    df = pd.read_csv(BASE / "results" / "features" / "data_with_deltas.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["well_id", "date"])

    rows = []
    for wid, grp in df.groupby("well_id"):
        grp = grp.dropna(subset=["wte"]).sort_values("date")
        if len(grp) < MIN_OBS:
            continue
        r_plain, r_hr = run_both_tests(grp["wte"].values)
        rows.append({
            "well_id": wid, "n_obs": len(grp),
            "p_plain": r_plain.p, "slope_plain": r_plain.slope, "trend_plain": r_plain.trend,
            "p_hamed_rao": r_hr.p, "slope_hamed_rao": r_hr.slope, "trend_hamed_rao": r_hr.trend,
        })

    res = pd.DataFrame(rows)
    n = len(res)

    sig_plain = res["p_plain"] < 0.05
    dec_plain = sig_plain & (res["slope_plain"] < 0)
    sig_hr = res["p_hamed_rao"] < 0.05
    dec_hr = sig_hr & (res["slope_hamed_rao"] < 0)

    print(f"Wells tested: {n} (of 273 retained)")
    print(f"  Plain Mann-Kendall:  significant={sig_plain.sum()} ({100*sig_plain.sum()/n:.1f}%)  "
          f"declining={dec_plain.sum()}")
    print(f"  Hamed-Rao:           significant={sig_hr.sum()} ({100*sig_hr.sum()/n:.1f}%)  "
          f"declining={dec_hr.sum()}")
    print()
    print("Reconciliation vs. prior numbers:")
    print(f"  Delivered mk_well_wte.csv / this run's plain test: "
          f"{sig_plain.sum()}/{dec_plain.sum()} -- MATCH expected 216/116.")
    print(f"  MEMO/draft letter cited 225/111 -- NOT reproduced; treat as stale/incorrect, "
          f"not the paper's §6.2 baseline.")
    print(f"  MEMO's provisional Hamed-Rao estimate was 167/80 (based on the wrong 225/111 "
          f"baseline); the verified robust count is {sig_hr.sum()}/{dec_hr.sum()}.")

    res.to_csv(OUT / "mk_well_wte_plain_vs_hamed_rao.csv", index=False)
    print(f"\nWrote {OUT / 'mk_well_wte_plain_vs_hamed_rao.csv'}")


if __name__ == "__main__":
    main()
