"""
Does the ΔWTE–ΔQ significance survive correct treatment of pseudo-replication?

The published per-gage regressions (Table 2) pool well-months and treat every row as
independent. They are not. Within a catchment,

    ΔQ(well w, month m) = q(m) − q0(w)

where q(m) is a SINGLE gage-level monthly discharge series shared by every well in the
catchment; wells differ only by the per-well constant q0(w). Bear River's n = 22,783 is
~99 wells x a few hundred shared months. The independent information is closer to the
number of months than to the number of rows, so a p-value computed at n = 22,783 is
inflated by construction.

This script re-estimates the same slopes four ways and reports them side by side:

  naive OLS        - reproduces the published Table 2 numbers (the baseline to beat)
  cluster by well  - allows arbitrary correlation within a well over time
  cluster by month - allows arbitrary correlation across wells within a gage-month;
                     this is the one that targets the shared q(m) directly
  two-way cluster  - Cameron-Gelbach-Miller: well + month - well&month
  catchment-month  - collapse to one (mean ΔWTE, mean ΔQ) point per gage-month, then
                     regress. n = number of months. This is also what Reviewer 1 asked
                     for in R1 #1, so the two requests share a fix.

Run:  ./.venv/bin/python notebooks/clustered_inference_check.py
      (or any python with pandas + statsmodels; it only reads a CSV)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).parent.parent
SRC = BASE / "result" / "features" / "data_with_deltas.csv"
OUT = BASE / "result" / "analysis" / "clustered_inference"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}


def two_way_cluster(y, X, g1, g2):
    """Cameron-Gelbach-Miller two-way clustered covariance."""
    m = sm.OLS(y, X)
    v1 = m.fit(cov_type="cluster", cov_kwds={"groups": g1}).cov_params()
    v2 = m.fit(cov_type="cluster", cov_kwds={"groups": g2}).cov_params()
    both = pd.Series(list(zip(g1, g2))).astype(str).values
    v12 = m.fit(cov_type="cluster", cov_kwds={"groups": both}).cov_params()
    V = np.asarray(v1) + np.asarray(v2) - np.asarray(v12)
    fit = m.fit()
    se = float(np.sqrt(np.diag(V)[1]))
    slope = float(np.asarray(fit.params)[1])
    # t with the conservative min(G1,G2)-1 dof
    dof = max(min(len(set(g1)), len(set(g2))) - 1, 1)
    from scipy import stats
    t = slope / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), dof) if np.isfinite(t) else np.nan
    return slope, se, t, p, dof


def fit_row(label, slope, se, t, p, n, note=""):
    return {"estimator": label, "slope": slope, "std_err": se,
            "t": t, "p_value": p, "n_units": n, "note": note}


def main():
    d = pd.read_csv(SRC, parse_dates=["date"], low_memory=False)
    d["gage_id"] = d["gage_id"].astype(str).str.replace(".0", "", regex=False)
    d = d.dropna(subset=["delta_wte", "delta_q"])
    d["ym"] = d["date"].dt.to_period("M").astype(str)

    rows = []
    for gid, g in d.groupby("gage_id"):
        if gid not in SHORT:
            continue
        name = SHORT[gid]
        y = g["delta_q"].to_numpy(dtype=float)
        X = sm.add_constant(g["delta_wte"].to_numpy(dtype=float))
        well, ym = g["well_id"].astype(str).values, g["ym"].values

        base = sm.OLS(y, X).fit()
        recs = [fit_row("naive OLS", base.params[1], base.bse[1], base.tvalues[1],
                        base.pvalues[1], len(g), "published method")]

        for lbl, grp in (("cluster: well", well), ("cluster: month", ym)):
            f = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": grp})
            recs.append(fit_row(lbl, f.params[1], f.bse[1], f.tvalues[1],
                                f.pvalues[1], len(set(grp))))

        s, se, t, p, dof = two_way_cluster(y, X, well, ym)
        recs.append(fit_row("cluster: well+month", s, se, t, p, dof + 1))

        agg = g.groupby("ym").agg(delta_wte=("delta_wte", "mean"),
                                  delta_q=("delta_q", "mean")).dropna()
        if len(agg) > 2:
            ya = agg["delta_q"].to_numpy(dtype=float)
            Xa = sm.add_constant(agg["delta_wte"].to_numpy(dtype=float))
            fa = sm.OLS(ya, Xa).fit()
            recs.append(fit_row("catchment-month mean", fa.params[1], fa.bse[1],
                                fa.tvalues[1], fa.pvalues[1], len(agg),
                                "= Reviewer 1 R1 #1"))

        for r in recs:
            r["gage"] = name
            r["gage_id"] = gid
        rows.extend(recs)

    res = pd.DataFrame(rows)[
        ["gage", "gage_id", "estimator", "slope", "std_err", "t", "p_value",
         "n_units", "note"]]
    res.to_csv(OUT / "clustered_inference_by_gage.csv", index=False)

    pd.set_option("display.width", 200)
    for name in [SHORT[k] for k in
                 ["10126000", "10141000", "10163000", "10152000", "10168000"]]:
        sub = res[res.gage == name]
        if not len(sub):
            continue
        print(f"\n=== {name} ===")
        for _, r in sub.iterrows():
            sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else \
                  "*" if r.p_value < 0.05 else "n.s."
            print(f"  {r.estimator:<22} slope {r.slope:>9.4f}  se {r.std_err:>9.4f}  "
                  f"p {r.p_value:>10.3g}  {sig:<4} n={int(r.n_units):>6,}  {r.note}")

    print("\n" + "=" * 78)
    print("SIGNIFICANCE AT p<0.05, BY ESTIMATOR")
    print("=" * 78)
    piv = res.pivot(index="gage", columns="estimator", values="p_value")
    order = ["naive OLS", "cluster: well", "cluster: month",
             "cluster: well+month", "catchment-month mean"]
    piv = piv[[c for c in order if c in piv.columns]]
    print((piv < 0.05).replace({True: "sig", False: "n.s."}).to_string())
    print("\np-values:")
    print(piv.map(lambda v: f"{v:.3g}").to_string())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
