"""
Negative control for the BFD conditioning (Reviewer 3, section A1).

Reviewer 3 argues the ΔWTE–ΔQ relationship is circular: baseflow-dominated months
are *selected* as those where streamflow tracks groundwater, so recovering a
positive slope there is guaranteed rather than evidence of coupling.

The test is to run the identical analysis on the months the classifier rejected.
If quickflow-affected months produce a comparable slope, the BFD conditioning is
doing no work and the circularity charge lands. If they separate, it does.

Two independent nulls are computed:

  1. Non-BFD control  — same pipeline, conditioned on bfd=0 instead of bfd=1.
  2. Permutation test — well→gage assignment shuffled, destroying the spatial
     pairing while preserving both marginal distributions.

Everything routes through the same gwbase functions the pipeline uses. The BFD
branch is produced here too rather than read from the published outputs, so the
two arms differ only in the conditioning, not in code path or environment.

Run with the pipeline venv, after a full pipeline run has populated results/:

    ./.venv/bin/python notebooks/negative_control_analysis.py

Writes results/analysis/negative_control/.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
import gwbase  # noqa: E402

RAW = BASE / "data" / "raw"
RESULTS = BASE / "results"
PROC = RESULTS / "processed"
OUT = RESULTS / "analysis" / "negative_control"
OUT.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 1000
SEED = 20260725  # fixed so the reported null is reproducible

GAGE_SHORT = {
    "10126000": "Bear River",
    "10141000": "Weber River",
    "10152000": "Spanish Fork",
    "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}


def build_arm(well_data, streamflow, use_bfd):
    """Run pairing → baseline → deltas for one conditioning arm.

    use_bfd=True  reproduces the published analysis (bfd=1 months).
    use_bfd=False is the negative control: the bfd flag is inverted before
    aggregation, so the same function selects quickflow-affected months and
    every downstream step is byte-identical in code path.
    """
    sf = streamflow.copy()
    if not use_bfd:
        sf["bfd"] = 1 - sf["bfd"]

    monthly = gwbase.aggregate_streamflow_monthly_bfd(sf)
    paired = gwbase.pair_wells_with_streamflow(well_data, monthly, None)
    paired = gwbase.calculate_baseline_values(paired)
    deltas = gwbase.compute_delta_metrics(paired)

    # Same datum-error exclusion the pipeline applies at Step 8. Applied to both
    # arms so the control is not advantaged or penalised by a known bad record.
    bad = ((deltas["gage_id"].astype(str) == "10163000")
           & (deltas["delta_wte"].abs() > 1400))
    if bad.any():
        deltas = deltas[~bad].copy()
    return deltas


def regress_by_gage(df, label):
    rows = []
    for gid, g in df.groupby(df["gage_id"].astype(str)):
        g = g.dropna(subset=["delta_wte", "delta_q"])
        if len(g) < 3:
            continue
        r = linregress(g["delta_wte"], g["delta_q"])
        rows.append({
            "arm": label,
            "gage_id": gid,
            "gage": GAGE_SHORT.get(gid, gid),
            "n_wells": g["well_id"].nunique(),
            "n_obs": len(g),
            "slope": r.slope,
            "intercept": r.intercept,
            "r_squared": r.rvalue ** 2,
            "p_value": r.pvalue,
            "std_err": r.stderr,
        })
    return pd.DataFrame(rows)


def pooled_slope(df):
    d = df.dropna(subset=["delta_wte", "delta_q"])
    if len(d) < 3:
        return np.nan
    return linregress(d["delta_wte"], d["delta_q"]).slope


def permutation_null(deltas, n_perm, seed):
    """Shuffle each well's gage label, breaking the spatial pairing.

    Preserves both marginal distributions and the within-well ΔWTE series, so
    the only thing destroyed is which catchment a well belongs to.
    """
    rng = np.random.default_rng(seed)
    well_gage = (deltas[["well_id", "gage_id"]].drop_duplicates()
                 .reset_index(drop=True))
    gages = well_gage["gage_id"].to_numpy()

    per_gage_null = {g: [] for g in set(deltas["gage_id"].astype(str))}
    pooled_null = []
    for i in range(n_perm):
        shuffled = well_gage.assign(gage_id=rng.permutation(gages))
        d = (deltas.drop(columns=["gage_id"])
             .merge(shuffled, on="well_id", how="left"))
        pooled_null.append(pooled_slope(d))
        for gid, g in d.groupby(d["gage_id"].astype(str)):
            per_gage_null.setdefault(gid, []).append(pooled_slope(g))
        if (i + 1) % 100 == 0:
            print(f"  permutation {i + 1}/{n_perm}")
    return np.array(pooled_null, dtype=float), per_gage_null


def main():
    print("Loading step-6 well data and raw streamflow...")
    well_data = pd.read_csv(PROC / "filtered_by_elevation.csv",
                            parse_dates=["date"])
    streamflow = gwbase.load_streamflow_data(
        str(RAW / "streamflow" / "gages_with_bfd_predictions"),
        filter_bfd=False,
    )

    # Same gage_id normalisation main_gwbase.py applies before Step 7. Without
    # it the well and streamflow frames carry int64 vs object keys and the
    # pairing merge fails.
    def norm(df):
        df = df.copy()
        df["gage_id"] = (df["gage_id"].astype(str)
                         .str.replace(".0", "", regex=False))
        return df

    well_data, streamflow = norm(well_data), norm(streamflow)
    keep = set(GAGE_SHORT)
    streamflow = streamflow[streamflow["gage_id"].isin(keep)].copy()
    well_data = well_data[well_data["gage_id"].isin(keep)].copy()

    print("\n=== ARM 1: BFD months (published conditioning) ===")
    bfd = build_arm(well_data, streamflow, use_bfd=True)
    print("\n=== ARM 2: non-BFD months (negative control) ===")
    nonbfd = build_arm(well_data, streamflow, use_bfd=False)

    reg = pd.concat([regress_by_gage(bfd, "bfd"),
                     regress_by_gage(nonbfd, "non_bfd")], ignore_index=True)
    reg.to_csv(OUT / "regression_by_arm.csv", index=False)

    print("\n" + "=" * 78)
    print("PER-GAGE SLOPES (cfs/ft)")
    print("=" * 78)
    wide = reg.pivot(index="gage", columns="arm",
                     values=["slope", "p_value", "n_obs"])
    print(wide.to_string())

    pooled = {"bfd": pooled_slope(bfd), "non_bfd": pooled_slope(nonbfd)}
    print(f"\nPooled slope, BFD months     : {pooled['bfd']:.4f} cfs/ft")
    print(f"Pooled slope, non-BFD months : {pooled['non_bfd']:.4f} cfs/ft")

    print(f"\n=== PERMUTATION TEST ({N_PERMUTATIONS} shuffles, seed {SEED}) ===")
    null_pooled, null_by_gage = permutation_null(bfd, N_PERMUTATIONS, SEED)
    obs = pooled["bfd"]
    p_emp = float((np.abs(null_pooled) >= abs(obs)).mean())
    print(f"\nobserved pooled slope : {obs:.4f}")
    print(f"null mean / sd        : {np.nanmean(null_pooled):.4f} / "
          f"{np.nanstd(null_pooled):.4f}")
    print(f"null 2.5–97.5 pct     : {np.nanpercentile(null_pooled, 2.5):.4f} "
          f"to {np.nanpercentile(null_pooled, 97.5):.4f}")
    print(f"empirical two-sided p : {p_emp:.4f}")

    pd.DataFrame({"permuted_pooled_slope": null_pooled}).to_csv(
        OUT / "permutation_null_pooled.csv", index=False)

    rows = []
    for gid, vals in null_by_gage.items():
        v = np.array(vals, dtype=float)
        o = reg[(reg.arm == "bfd") & (reg.gage_id == gid)]["slope"]
        if not len(o) or not np.isfinite(v).any():
            continue
        o = float(o.iloc[0])
        rows.append({
            "gage_id": gid, "gage": GAGE_SHORT.get(gid, gid),
            "observed_slope": o,
            "null_mean": np.nanmean(v), "null_sd": np.nanstd(v),
            "null_p2.5": np.nanpercentile(v, 2.5),
            "null_p97.5": np.nanpercentile(v, 97.5),
            "empirical_p": float((np.abs(v) >= abs(o)).mean()),
        })
    per_gage = pd.DataFrame(rows).sort_values("gage")
    per_gage.to_csv(OUT / "permutation_by_gage.csv", index=False)
    print("\nPER-GAGE PERMUTATION RESULTS")
    print(per_gage.to_string(index=False))

    summary = pd.DataFrame([{
        "pooled_slope_bfd": pooled["bfd"],
        "pooled_slope_non_bfd": pooled["non_bfd"],
        "n_obs_bfd": len(bfd), "n_obs_non_bfd": len(nonbfd),
        "permutation_n": N_PERMUTATIONS, "permutation_seed": SEED,
        "permutation_null_mean": float(np.nanmean(null_pooled)),
        "permutation_null_sd": float(np.nanstd(null_pooled)),
        "permutation_empirical_p": p_emp,
    }])
    summary.to_csv(OUT / "summary.csv", index=False)
    print(f"\nWrote {OUT}")
    print("\nINTERPRETATION IS NOT AUTOMATED. Report what these numbers show, "
          "including if the control fails to separate from the BFD arm.")


if __name__ == "__main__":
    main()
