"""
Does the lagged coupling survive correct inference?

The published lag analysis (Section 5.7, Figure 16) reports dR^2 and dMI at 3mo, 6mo, 1yr
and 5yr lags, and finds both negligible. But the per-gage SLOPES at those lags are not in
the paper, and they are striking: at a 6-month lag four of five gages turn positive and
"significant" under pooled OLS.

Those p-values carry the same pseudo-replication inflation as Table 2, so before anyone
builds an argument on them they need re-estimating with:

  well fixed effects   -- removes the between-well component that hijacks pooled fits
  clustered std errors -- error bars from wells, not rows

This matters beyond the statistics. A lagged peak is an argument against Reviewer 3's
circularity charge: selection on contemporaneous flow would produce a contemporaneous
artifact, not one peaking half a year later. But only if the lagged result is real.

Run:  ./.venv/bin/python notebooks/lag_robust_inference.py
"""
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).parent.parent
FEAT = BASE / "result" / "features"
OUT = BASE / "result" / "analysis" / "lag_robust"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}
LAGS = [
    ("no lag",  "data_with_deltas.csv", "delta_wte"),
    ("3 month", "data_lag_3mo.csv",     "delta_wte_lag_3_months"),
    ("6 month", "data_lag_6mo.csv",     "delta_wte_lag_6_months"),
    ("1 year",  "data_lag_1yr.csv",     "delta_wte_lag_1_year"),
    ("5 year",  "data_lag_5yr.csv",     "delta_wte_lag_5_years"),
]


def main():
    rows = []
    for label, fname, xcol in LAGS:
        f = FEAT / fname
        if not f.is_file():
            print(f"skip {label}: {fname} missing")
            continue
        d = pd.read_csv(f, low_memory=False)
        if xcol not in d.columns:
            cand = [c for c in d.columns if "lag" in c.lower()]
            if not cand:
                print(f"skip {label}: no lag column in {fname}")
                continue
            xcol = cand[0]
        d["gage_id"] = d.gage_id.astype(str).str.replace(".0", "", regex=False)
        d["well_id"] = d.well_id.astype(str)
        d = d.dropna(subset=[xcol, "delta_q"])

        for gid, g in d.groupby("gage_id"):
            if gid not in SHORT or g.well_id.nunique() < 3:
                continue
            y = g.delta_q.to_numpy(float)
            x = g[xcol].to_numpy(float)

            pooled = sm.OLS(y, sm.add_constant(x)).fit()

            gg = g.copy()
            gg["xd"] = gg[xcol] - gg.groupby("well_id")[xcol].transform("mean")
            gg["yd"] = gg.delta_q - gg.groupby("well_id").delta_q.transform("mean")
            if gg.xd.std() == 0:
                continue
            fe = sm.OLS(gg.yd.to_numpy(float), gg.xd.to_numpy(float)).fit(
                cov_type="cluster", cov_kwds={"groups": gg.well_id.values})

            rows.append({
                "lag": label, "gage": SHORT[gid],
                "pooled_slope": pooled.params[1], "pooled_p": pooled.pvalues[1],
                "fe_slope": fe.params[0], "fe_p": fe.pvalues[0],
                "n_obs": len(g), "n_wells": g.well_id.nunique(),
            })

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "lag_robust_by_gage.csv", index=False)
    order = [l[0] for l in LAGS]
    gorder = ["Bear River", "Weber River", "Provo River", "Spanish Fork",
              "Little Cottonwood"]

    def show(title, scol, pcol):
        print(f"\n=== {title} ===")
        piv = r.pivot(index="gage", columns="lag", values=scol)
        piv = piv.reindex(gorder)[[c for c in order if c in piv.columns]]
        pp = r.pivot(index="gage", columns="lag", values=pcol)
        pp = pp.reindex(gorder)[[c for c in order if c in pp.columns]]
        hdr = "".join(f"{c:>16}" for c in piv.columns)
        print(f"{'gage':<19}{hdr}")
        for gname in piv.index:
            cells = ""
            for c in piv.columns:
                s, p = piv.loc[gname, c], pp.loc[gname, c]
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                cells += f"{f'{s:.3f}{star}':>16}"
            print(f"{gname:<19}{cells}")

    show("POOLED OLS slope (as published; p-values inflated)", "pooled_slope", "pooled_p")
    show("WELL FIXED EFFECTS slope, clustered p", "fe_slope", "fe_p")

    print("\n=== SIGNIFICANT AT p<0.05, COUNT OF GAGES ===")
    print(f"{'lag':<12}{'pooled OLS':>14}{'fixed effects':>16}")
    for l in order:
        sub = r[r.lag == l]
        if not len(sub):
            continue
        print(f"{l:<12}{(sub.pooled_p<0.05).sum():>10} / {len(sub)}"
              f"{(sub.fe_p<0.05).sum():>12} / {len(sub)}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
