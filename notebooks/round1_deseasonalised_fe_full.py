"""
Round-1 revision, item 1 — Deseasonalised within-well spec, finalized.

Extends notebooks/deseasonalised_fe_lag.py with what the response letters still
need before the Section 5 table can be finalized:

  - within-well R^2 (on the demeaned/within-transformed data), not just slope + p
  - the normalized (fractional, dQ/Q0) fixed-effects slope alongside the
    absolute (cfs/ft) one
  - explicit 95% CI (slope +/- t_crit * clustered SE), not just a p-value
  - reported for all five gages, with the four retained under the new
    ten-well catchment threshold (Bear River, Weber River, Provo River,
    Little Cottonwood) flagged separately from Spanish Fork (7 wells, dropped
    by rule)

Same deseasonalisation as the original script: subtract month-of-year means,
per well for dWTE, per gage for dQ. Same estimator: well fixed effects
(within transform), clustered by well.

Source data: results/features/data_with_deltas.csv (the delivered/verified
data — reproduces the published Table 2 to floating-point precision, see
HANDOFF.md, so it is equivalent to a fresh `result/` pipeline run for this
purpose).

Run:  ./.venv/bin/python notebooks/round1_deseasonalised_fe_full.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE = Path(__file__).parent.parent
SRC = BASE / "results" / "features" / "data_with_deltas.csv"
OUT = BASE / "results" / "round1_revision" / "01_deseasonalised_within_well"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}
RETAINED = {"Bear River", "Weber River", "Provo River", "Little Cottonwood"}
LAGS = [0, 3, 6, 12]


def lag_by_date(g, col, months):
    """True calendar lag (merge on well_id + date offset), not a row shift."""
    if months == 0:
        return g[col]
    left = g[["well_id", "date"]].copy()
    left["key"] = left["date"] - pd.DateOffset(months=months)
    right = g[["well_id", "date", col]].rename(columns={"date": "key", col: "_lagged"})
    merged = left.merge(right, on=["well_id", "key"], how="left")
    return merged["_lagged"].to_numpy()


def fe_fit(g, xcol, ycol):
    """Well fixed effects via the within transform, clustered by well.

    Returns slope, clustered SE, p, 95% CI, within-R^2, and n.
    Within-R^2 = 1 - SS_res / SS_tot, computed on the demeaned data (the
    standard fixed-effects convention) — NOT statsmodels' default uncentered
    R^2, which would use sum(y_demeaned^2) as if the mean were still zero.
    """
    gg = g.dropna(subset=[xcol, ycol]).copy()
    n_wells = gg.well_id.nunique()
    if gg[xcol].std() == 0 or n_wells < 3 or len(gg) < 5:
        return dict(slope=np.nan, se=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    r2=np.nan, n_obs=len(gg), n_wells=n_wells, n_clusters=n_wells)

    gg["xd"] = gg[xcol] - gg.groupby("well_id")[xcol].transform("mean")
    gg["yd"] = gg[ycol] - gg.groupby("well_id")[ycol].transform("mean")
    y = gg.yd.to_numpy(float)
    x = gg.xd.to_numpy(float)
    m = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": gg.well_id.values})

    slope = float(m.params[0])
    se = float(m.bse[0])
    p = float(m.pvalues[0])
    dof = max(n_wells - 1, 1)  # conservative: clusters - 1
    tcrit = stats.t.ppf(0.975, dof)
    ci_lo, ci_hi = slope - tcrit * se, slope + tcrit * se

    resid = y - slope * x
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return dict(slope=slope, se=se, p=p, ci_lo=ci_lo, ci_hi=ci_hi, r2=r2,
                n_obs=len(gg), n_wells=n_wells, n_clusters=n_wells)


def main():
    d = pd.read_csv(SRC, parse_dates=["date"], low_memory=False)
    d["gage_id"] = d.gage_id.astype(str).str.replace(".0", "", regex=False)
    d["well_id"] = d.well_id.astype(str)
    d = d.dropna(subset=["delta_wte", "delta_q"])
    d = d[d.gage_id.isin(SHORT)].copy()
    d = d[d.q0 > 0].copy()
    d["mon"] = d.date.dt.month
    d["delta_q_norm"] = d.delta_q / d.q0

    # Deseasonalise: subtract month-of-year means, per well for dWTE (each well
    # has its own seasonal amplitude), per gage for dQ (one hydrograph per
    # catchment). Fractional dQ is deseasonalised the same way, per gage.
    d["w_ds"] = d.delta_wte - d.groupby(["well_id", "mon"]).delta_wte.transform("mean")
    d["q_ds"] = d.delta_q - d.groupby(["gage_id", "mon"]).delta_q.transform("mean")
    d["qn_ds"] = d.delta_q_norm - d.groupby(["gage_id", "mon"]).delta_q_norm.transform("mean")
    d = d.sort_values(["well_id", "date"])

    rows = []
    for gid, g in d.groupby("gage_id"):
        gname = SHORT[gid]
        for L in LAGS:
            gg = g.sort_values(["well_id", "date"]).reset_index(drop=True)
            gg["xlag_raw"] = lag_by_date(gg, "delta_wte", L)
            gg["xlag_ds"] = lag_by_date(gg, "w_ds", L)

            specs = [
                ("raw", "xlag_raw", "delta_q"),
                ("deseasonalised", "xlag_ds", "q_ds"),
                ("deseasonalised_normalized", "xlag_ds", "qn_ds"),
            ]
            for label, xc, yc in specs:
                fit = fe_fit(gg, xc, yc)
                rows.append({"gage": gname, "gage_id": gid, "retained": gname in RETAINED,
                             "lag_months": L, "series": label, **fit})

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "deseasonalised_fe_full.csv", index=False)

    def stars(p):
        return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s.")

    print("=" * 100)
    print("ZERO-LAG RESULTS — the featured Section 5 specification")
    print("=" * 100)
    z = r[(r.lag_months == 0) & (r.series == "deseasonalised")].sort_values(
        "retained", ascending=False)
    for _, row in z.iterrows():
        flag = "" if row.retained else "  [EXCLUDED, <10 wells]"
        print(f"{row.gage:<19} slope {row.slope:+7.3f} cfs/ft  "
              f"95% CI [{row.ci_lo:+7.3f}, {row.ci_hi:+7.3f}]  "
              f"p={row.p:.4g} {stars(row.p):<5} R^2={row.r2:.4f}  "
              f"n={int(row.n_obs):,} ({int(row.n_wells)} wells){flag}")

    print("\n" + "=" * 100)
    print("ZERO-LAG NORMALIZED (FRACTIONAL, dQ/Q0) RESULTS")
    print("=" * 100)
    zn = r[(r.lag_months == 0) & (r.series == "deseasonalised_normalized")].sort_values(
        "retained", ascending=False)
    for _, row in zn.iterrows():
        flag = "" if row.retained else "  [EXCLUDED, <10 wells]"
        print(f"{row.gage:<19} slope {row.slope:+9.5f} ft^-1  "
              f"95% CI [{row.ci_lo:+9.5f}, {row.ci_hi:+9.5f}]  "
              f"p={row.p:.4g} {stars(row.p):<5} R^2={row.r2:.4f}{flag}")

    print("\n" + "=" * 100)
    print("RAW (NOT deseasonalised) within-well fit, for comparison — Response 1(b)/(c)")
    print("=" * 100)
    zr = r[(r.lag_months == 0) & (r.series == "raw")].sort_values("retained", ascending=False)
    for _, row in zr.iterrows():
        flag = "" if row.retained else "  [EXCLUDED, <10 wells]"
        print(f"{row.gage:<19} slope {row.slope:+7.3f} cfs/ft  "
              f"95% CI [{row.ci_lo:+7.3f}, {row.ci_hi:+7.3f}]  "
              f"p={row.p:.4g} {stars(row.p):<5} R^2={row.r2:.4f}  "
              f"n={int(row.n_obs):,} ({int(row.n_wells)} wells){flag}")

    print("\n" + "=" * 100)
    print("LAG PROFILE (deseasonalised, absolute cfs/ft)")
    print("=" * 100)
    sub = r[r.series == "deseasonalised"]
    piv = sub.pivot(index="gage", columns="lag_months", values="slope")
    pp = sub.pivot(index="gage", columns="lag_months", values="p")
    print(f"{'gage':<19}" + "".join(f"{f'{L} mo':>14}" for L in LAGS))
    for gname in piv.index:
        cells = ""
        for L in LAGS:
            s, p = piv.loc[gname, L], pp.loc[gname, L]
            star = stars(p) if np.isfinite(p) else ""
            cells += f"{f'{s:.3f}{star}':>14}" if np.isfinite(s) else f"{'--':>14}"
        print(f"{gname:<19}{cells}")

    ds0 = r[(r.series == "deseasonalised") & (r.lag_months == 0) & (r.retained)]
    print(f"\nAt zero lag, among the 4 retained gages: {(ds0.slope > 0).sum()} of "
          f"{len(ds0)} positive, {(ds0.p < 0.05).sum()} significant at p<0.05.")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
