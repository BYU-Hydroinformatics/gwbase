"""
Round-1 revision, item 7 -- Descriptive statistics table (R3 B19).

Reviewer 3 (p. 21): "What are the statistical distribution of flows and
water levels for each catchment?" -- a reasonable request the manuscript
should have anticipated, so readers can assess the distributions underlying
every regression rather than inferring them from scatter plots.

Produces n, mean, median, interquartile range (Q1-Q3), and full range
(min-max) of both discharge (Q, cfs) and water-table elevation (WTE, ft)
for each terminal-gage catchment, from the same paired well-month records
(results/features/data_with_deltas.csv) used throughout Section 5. Spanish
Fork is included and flagged (excluded from catchment rankings/basin-scale
figures elsewhere by the ten-well minimum, but its own distribution is
still meaningful to report here).

Run:  ./.venv/bin/python notebooks/round1_descriptive_stats_table.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent.parent
SRC = BASE / "results" / "features" / "data_with_deltas.csv"
OUT = BASE / "results" / "round1_revision" / "07_descriptive_stats_table"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "10126000": "Bear River", "10141000": "Weber River",
    "10152000": "Spanish Fork", "10163000": "Provo River",
    "10168000": "Little Cottonwood",
}
RETAINED = ["Bear River", "Weber River", "Provo River", "Little Cottonwood", "Spanish Fork"]


def describe(x: pd.Series) -> dict:
    x = x.dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    return {
        "n": len(x), "mean": x.mean(), "median": x.median(),
        "q1": q1, "q3": q3, "iqr": q3 - q1,
        "min": x.min(), "max": x.max(),
    }


def main():
    d = pd.read_csv(SRC, low_memory=False)
    d["gage_id"] = d["gage_id"].astype(str).str.replace(".0", "", regex=False)
    d = d[d["gage_id"].isin(SHORT)].copy()
    d["gage"] = d["gage_id"].map(SHORT)

    rows = []
    for gname, g in d.groupby("gage"):
        for var, col in [("Q", "q"), ("WTE", "wte")]:
            stats = describe(g[col])
            rows.append({"gage": gname, "variable": var, **stats})

    out = pd.DataFrame(rows).set_index("gage").loc[RETAINED].reset_index()
    out.to_csv(OUT / "descriptive_stats_by_gage.csv", index=False)

    pd.set_option("display.width", 160)
    print("=" * 100)
    print("R3 B19 -- DESCRIPTIVE STATISTICS BY TERMINAL-GAGE CATCHMENT")
    print("=" * 100)
    for var in ["Q", "WTE"]:
        unit = "cfs" if var == "Q" else "ft"
        print(f"\n--- {var} ({unit}) ---")
        sub = out[out.variable == var]
        for _, row in sub.iterrows():
            flag = "  [<10 wells, excluded from rankings]" if row.gage == "Spanish Fork" else ""
            print(f"  {row.gage:<19} n={int(row.n):6,}  mean={row['mean']:9.2f}  "
                  f"median={row['median']:9.2f}  IQR=[{row.q1:8.2f}, {row.q3:8.2f}]  "
                  f"range=[{row['min']:9.2f}, {row['max']:9.2f}]{flag}")

    print(f"\nWrote {OUT / 'descriptive_stats_by_gage.csv'}")


if __name__ == "__main__":
    main()
