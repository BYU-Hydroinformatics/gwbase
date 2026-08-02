"""
Example streamflow hydrograph with baseflow-dominated (BFD) days highlighted
(manuscript Figure 5, label fig:bfd).

The original figure-generating script could not be located (not in
visualization.py, notebooks/*.py, git history, or the paper repo) -- this is
a from-scratch replacement, not a byte-for-byte recreation of the original.
Gage: Little Cottonwood Creek (10168000), 2015-01-01 to 2018-01-01. Chosen
for a wide dynamic range that exercises the linear-vs-log axis question
(R3 B5), and because it is the paper's headline most-groundwater-sensitive
catchment (Section 6.1). Blue (BFD=1) points landing on snowmelt spikes are
not a plotting error -- they reflect the classifier's known over-inclusion
at storm/recession onset, the same behavior R3's A2 comment cites Figure 5
as demonstrating (see reviewer_3_response.md A2/B4).

Produces both a linear and a log-scale y-axis version so the two can be
compared before deciding which goes in the manuscript -- njones flagged that
a log axis expands the low-flow range and may emphasize small baseflow
excursions rather than defuse Reviewer 3's "spikes" critique, so this is a
judgment call, not a mechanical plt.yscale('log') swap.

Run:  ./.venv/bin/python notebooks/plot_bfd_example_hydrograph.py
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = Path(__file__).parent.parent
OUT_DIR = BASE / "results" / "analysis" / "filter_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAGE_ID = "10168000"
START, END = "2015-01-01", "2018-01-01"


def make_plot(df, yscale, out_path):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df["date"], df["streamflow"], color="black", linewidth=0.7,
            label="Discharge", zorder=2)
    bfd = df[df["bfd"] == 1]
    ax.scatter(bfd["date"], bfd["streamflow"], color="blue", s=8,
               edgecolors="none", label="Baseflow", zorder=3)
    ax.set_yscale(yscale)
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (cfs)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    df = pd.read_csv(BASE / "data" / "raw" / "streamflow" /
                      "gages_with_bfd_predictions" / f"{GAGE_ID}.csv",
                      parse_dates=["date"])
    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()

    make_plot(df, "linear", OUT_DIR / "bfd_example_hydrograph_linear.png")
    make_plot(df, "log", OUT_DIR / "bfd_example_hydrograph_log.png")


if __name__ == "__main__":
    main()
