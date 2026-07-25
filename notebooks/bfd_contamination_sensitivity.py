"""
Do the ΔWTE–ΔQ results depend on BFD misclassification? (Reviewer 3, part B)

Reviewer 3 argues the random-forest classifier captures early-rain spikes as
baseflow, and — crucially — that "these outliers THEN RESULT IN the observed very
tenuous trends in all subsequent steps."

That is a causal claim, so it is falsifiable, and it does not require showing the
classifier is perfect. No classifier is. It requires showing the RESULTS DO NOT
DEPEND on its errors. So: aggressively trim every BFD day that could plausibly be
storm-affected, and see whether the slopes move.

Trim levels, increasingly severe:

  none          published BFD set (baseline to compare against)
  rise>50%      drop BFD days whose discharge rose >50% over the previous day
  rise>20%      same at a stricter threshold
  rise>20%+3d   also drop the 3 days FOLLOWING such a rise (the recession limb of
                a storm, which a classifier would plausibly still label baseflow)
  low-half      keep only BFD days at or below the median BFD-day flow for that
                gage -- deliberately brutal, discards half the data

IMPORTANT — the baseline is held FIXED at the published (untrimmed) wte0/q0.
Trimming can remove a well's earliest BFD month, which would shift its Method 1
baseline and confound "contamination effect" with "baseline effect". Those are
different things and an earlier analysis in this project conflated them.

Run:  ./.venv/bin/python notebooks/bfd_contamination_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
import gwbase  # noqa: E402

RAW = BASE / "data" / "raw"
PROC = BASE / "result" / "processed"
OUT = BASE / "result" / "analysis" / "bfd_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}


def norm(df):
    df = df.copy()
    df["gage_id"] = df["gage_id"].astype(str).str.replace(".0", "", regex=False)
    return df


def trim_masks(sf):
    """Boolean masks over BFD days, marking those to KEEP under each trim level."""
    sf = sf.sort_values(["gage_id", "date"]).copy()
    prev = sf.groupby("gage_id")["q"].shift(1)
    prev = prev.where(prev > 0)          # zero/negative prior flow -> undefined ratio
    sf["rise"] = sf["q"].astype(float) / prev.astype(float) - 1.0

    spike20 = (sf["rise"] > 0.20).fillna(False)
    after = spike20.copy()
    for k in (1, 2, 3):  # recession limb following a spike, within the same gage
        shifted = spike20.groupby(sf["gage_id"]).shift(k).fillna(False)
        after = after | shifted.astype(bool)

    med = sf[sf.bfd == 1].groupby("gage_id")["q"].transform("median")
    med_full = pd.Series(np.nan, index=sf.index)
    med_full.loc[med.index] = med
    med_full = med_full.groupby(sf["gage_id"]).transform(
        lambda s: s.ffill().bfill())

    return sf, {
        "none":        pd.Series(True, index=sf.index),
        "rise>50%":    ~(sf["rise"] > 0.50).fillna(False),
        "rise>20%":    ~spike20,
        "rise>20%+3d": ~after,
        "low-half":    sf["q"] <= med_full,
    }


def run_variant(sf, keep, well_data, baseline):
    """Aggregate -> pair -> deltas with a FIXED baseline; regress per gage."""
    s = sf.copy()
    s.loc[~keep, "bfd"] = 0                      # demote trimmed days
    monthly = gwbase.aggregate_streamflow_monthly_bfd(s)
    paired = gwbase.pair_wells_with_streamflow(well_data, monthly, None)

    paired = paired.merge(baseline, on="well_id", how="left")
    paired["delta_wte"] = pd.to_numeric(paired["wte"], errors="raise") - paired["wte0"]
    paired["delta_q"] = pd.to_numeric(paired["q"], errors="raise") - paired["q0"]

    bad = ((paired["gage_id"].astype(str) == "10163000")
           & (paired["delta_wte"].abs() > 1400))
    paired = paired[~bad]

    rows = []
    for gid, g in paired.groupby(paired["gage_id"].astype(str)):
        g = g.dropna(subset=["delta_wte", "delta_q"])
        if gid not in SHORT or len(g) < 10:
            continue
        r = linregress(g["delta_wte"], g["delta_q"])
        rows.append({"gage": SHORT[gid], "slope": r.slope, "r2": r.rvalue ** 2,
                     "p": r.pvalue, "n_obs": len(g),
                     "n_wells": g["well_id"].nunique()})
    return pd.DataFrame(rows)


def main():
    well_data = norm(pd.read_csv(PROC / "filtered_by_elevation.csv",
                                 parse_dates=["date"]))
    sf = norm(gwbase.load_streamflow_data(
        str(RAW / "streamflow" / "gages_with_bfd_predictions"), filter_bfd=False))
    keepg = set(SHORT)
    sf = sf[sf.gage_id.isin(keepg)].copy()
    well_data = well_data[well_data.gage_id.isin(keepg)].copy()

    # Published baseline, computed once on the untrimmed data and reused throughout.
    monthly0 = gwbase.aggregate_streamflow_monthly_bfd(sf)
    paired0 = gwbase.calculate_baseline_values(
        gwbase.pair_wells_with_streamflow(well_data, monthly0, None))
    baseline = (paired0[["well_id", "wte0", "q0"]].drop_duplicates("well_id")
                .reset_index(drop=True))
    print(f"\nFixed baseline captured for {len(baseline):,} wells\n")

    sf, masks = trim_masks(sf)
    n_bfd = int((sf.bfd == 1).sum())

    all_res = []
    for label, keep in masks.items():
        dropped = int(((sf.bfd == 1) & ~keep).sum())
        print(f"--- trim '{label}': dropping {dropped:,} of {n_bfd:,} BFD days "
              f"({dropped / n_bfd * 100:.1f}%)")
        res = run_variant(sf, keep, well_data, baseline)
        res["trim"] = label
        res["bfd_days_dropped_pct"] = dropped / n_bfd * 100
        all_res.append(res)

    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(OUT / "bfd_trim_sensitivity.csv", index=False)

    print("\n" + "=" * 92)
    print("SLOPE (cfs/ft) BY TRIM LEVEL — does the result depend on possible contamination?")
    print("=" * 92)
    piv = res.pivot(index="gage", columns="trim", values="slope")
    piv = piv[[c for c in masks if c in piv.columns]]
    print(piv.round(4).to_string())
    print("\nn_obs retained:")
    print(res.pivot(index="gage", columns="trim", values="n_obs")[
        [c for c in masks if c in piv.columns]].to_string())
    print("\n% change in slope vs untrimmed:")
    rel = piv.div(piv["none"], axis=0).sub(1).mul(100)
    print(rel.round(1).to_string())
    print(f"\nWrote {OUT}")
    print("\nIf slopes are stable across trim levels, Reviewer 3's causal claim "
          "-- that misclassified spikes PRODUCE the trend -- is refuted, whether "
          "or not the classifier is perfect.")


if __name__ == "__main__":
    main()
