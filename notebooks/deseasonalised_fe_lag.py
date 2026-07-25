"""
Deseasonalised dWTE-dQ coupling, estimated within wells (the corrected Method 1).

Two confounds sit on top of the published Method 1 result and they interact:

  1. SEASONALITY. Method 1 deltas retain the annual cycle -- Reviewer 1's objection.
     For two annually cycling series a 6-month lag aligns OPPOSITE seasons, which flips
     the correlation sign, and a 12-month lag realigns them. So the raw lag profile
     oscillates in sign with an annual period, and any "peak" at 6 months may be phase
     alignment rather than subsurface travel time. Bear River shows exactly this:
     -2.01, +3.21, +5.02, -5.10, -5.47 across 0/3/6/12/60 months.

  2. POOLING. Pooled OLS lets between-well baseline differences drive the slope
     (see clustered_inference_check.py). Well fixed effects removes that.

This script removes both. Seasonality is stripped by subtracting month-of-year means --
per well for dWTE, per gage for dQ -- and the slope is estimated with well fixed effects
and well-clustered standard errors.

Result: the sign oscillation disappears, and ALL FIVE gages turn positive at zero lag with
three significant, against one significant and two negative in the published Table 2. The
apparent 6-month peak flattens into a 0-6 month plateau, so the honest reading is
contemporaneous-to-seasonal coupling, not a sharp travel-time signature.

Run:  ./.venv/bin/python notebooks/deseasonalised_fe_lag.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).parent.parent
SRC = BASE / "result" / "features" / "data_with_deltas.csv"
OUT = BASE / "result" / "analysis" / "deseasonalised_fe"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}
LAGS = [0, 3, 6, 12]


def lag_by_date(g, col, months):
    """True calendar lag, matching the pipeline's create_lag_analysis.

    A groupby.shift(n) moves n ROWS, which only equals n months if every well's
    monthly series is gap-free. It is not, so shift silently mislabels the lag.
    """
    if months == 0:
        return g[col]
    left = g[["well_id", "date"]].copy()
    left["key"] = left["date"] - pd.DateOffset(months=months)
    right = g[["well_id", "date", col]].rename(
        columns={"date": "key", col: "_lagged"})
    merged = left.merge(right, on=["well_id", "key"], how="left")
    return merged["_lagged"].to_numpy()


def fe_slope(g, xcol, ycol):
    """Well fixed effects via the within transform, clustered by well."""
    gg = g.dropna(subset=[xcol, ycol]).copy()
    if gg[xcol].std() == 0 or gg.well_id.nunique() < 3:
        return np.nan, np.nan, 0
    gg["xd"] = gg[xcol] - gg.groupby("well_id")[xcol].transform("mean")
    gg["yd"] = gg[ycol] - gg.groupby("well_id")[ycol].transform("mean")
    m = sm.OLS(gg.yd.to_numpy(float), gg.xd.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": gg.well_id.values})
    return m.params[0], m.pvalues[0], len(gg)


def main():
    d = pd.read_csv(SRC, parse_dates=["date"], low_memory=False)
    d["gage_id"] = d.gage_id.astype(str).str.replace(".0", "", regex=False)
    d["well_id"] = d.well_id.astype(str)
    d = d.dropna(subset=["delta_wte", "delta_q"])
    d = d[d.gage_id.isin(SHORT)].copy()
    d["mon"] = d.date.dt.month

    # Remove the month-of-year cycle. Per well for the water table (each well has its
    # own seasonal amplitude); per gage for discharge (one hydrograph per catchment).
    d["w_ds"] = d.delta_wte - d.groupby(["well_id", "mon"]).delta_wte.transform("mean")
    d["q_ds"] = d.delta_q - d.groupby(["gage_id", "mon"]).delta_q.transform("mean")
    d = d.sort_values(["well_id", "date"])

    rows = []
    for gid, g in d.groupby("gage_id"):
        for L in LAGS:
            gg = g.copy()
            gg = gg.sort_values(["well_id", "date"]).reset_index(drop=True)
            gg["xlag"] = lag_by_date(gg, "w_ds", L)
            for label, xc, yc in (("raw", "delta_wte", "delta_q"),
                                  ("deseasonalised", "xlag", "q_ds")):
                if label == "raw":
                    gg["xraw"] = lag_by_date(gg, "delta_wte", L)
                    xc = "xraw"
                s, p, n = fe_slope(gg, xc, yc)
                rows.append({"gage": SHORT[gid], "lag_months": L, "series": label,
                             "slope": s, "p_value": p, "n_obs": n,
                             "n_wells": g.well_id.nunique()})

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "deseasonalised_fe_lag.csv", index=False)

    for series in ("raw", "deseasonalised"):
        print(f"\n=== {series.upper()} — well fixed effects, clustered errors ===")
        sub = r[r.series == series]
        piv = sub.pivot(index="gage", columns="lag_months", values="slope")
        pp = sub.pivot(index="gage", columns="lag_months", values="p_value")
        print(f"{'gage':<19}" + "".join(f"{f'{L} mo':>14}" for L in LAGS))
        for gname in piv.index:
            cells = ""
            for L in LAGS:
                s, p = piv.loc[gname, L], pp.loc[gname, L]
                star = ("***" if p < 0.001 else "**" if p < 0.01
                        else "*" if p < 0.05 else "")
                cells += f"{f'{s:.3f}{star}':>14}" if np.isfinite(s) else f"{'--':>14}"
            print(f"{gname:<19}{cells}")

    ds0 = r[(r.series == "deseasonalised") & (r.lag_months == 0)]
    print(f"\nAt zero lag, deseasonalised: {(ds0.slope > 0).sum()} of {len(ds0)} gages "
          f"positive, {(ds0.p_value < 0.05).sum()} significant at p<0.05.")
    print("Published Table 2 for comparison: 1 of 5 significant, 2 negative.")
    print("\n* p<0.05  ** p<0.01  *** p<0.001")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
