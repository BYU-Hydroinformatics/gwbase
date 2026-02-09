"""
Well data quality summary module.

Produces concise statistics and figures characterizing groundwater well data:
- Information content (n_obs)
- Temporal coverage (record_length_years)
- Effective temporal resolution (median_sampling_interval_days)

Uses raw observation data only; no interpolation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def compute_well_summary_metrics(
    df: pd.DataFrame,
    well_id_col: str = "well_id",
    date_col: str = "datetime",
) -> pd.DataFrame:
    """
    Compute per-well summary metrics from raw observation data.

    Parameters
    ----------
    df : pd.DataFrame
        One row per measurement. Must contain well_id, datetime, wte (or depth).
    well_id_col : str
        Column name for well identifier.
    date_col : str
        Column name for observation date/datetime.

    Returns
    -------
    pd.DataFrame
        One row per well with columns:
        well_id, n_obs, record_length_years, median_sampling_interval_days
    """
    if well_id_col not in df.columns or date_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{well_id_col}' and '{date_col}' columns")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    rows = []
    for wid, grp in df.groupby(well_id_col):
        dates = grp[date_col].dropna().sort_values().values
        n_obs = len(dates)

        if n_obs == 0:
            rows.append({
                well_id_col: wid,
                "n_obs": 0,
                "record_length_years": 0.0,
                "median_sampling_interval_days": np.nan,
            })
            continue

        start_date = pd.Timestamp(dates[0])
        end_date = pd.Timestamp(dates[-1])
        td = end_date - start_date
        record_length_days = td.total_seconds() / 86400.0
        record_length_years = record_length_days / 365.25 if record_length_days > 0 else 0.0

        if n_obs >= 2:
            dts = pd.to_datetime(dates).values
            diffs = np.diff(dts) / np.timedelta64(1, "D")
            median_interval = float(np.median(diffs))
        else:
            median_interval = np.nan

        rows.append({
            well_id_col: wid,
            "n_obs": n_obs,
            "record_length_years": record_length_years,
            "median_sampling_interval_days": median_interval,
        })

    return pd.DataFrame(rows)


def compute_global_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global summary statistics across all wells.

    Returns
    -------
    pd.DataFrame
        One row with statistic, value columns.
    """
    n_wells = len(metrics)
    n_obs = metrics["n_obs"]
    med_int = metrics["median_sampling_interval_days"]

    median_n_obs = float(n_obs.median())
    p75_n_obs = float(n_obs.quantile(0.75))
    max_n_obs = int(n_obs.max())
    median_record_years = float(metrics["record_length_years"].median())
    pct_n_obs_le2 = 100.0 * (n_obs <= 2).sum() / n_wells
    pct_med_int_le30 = 100.0 * med_int.dropna().le(30).sum() / n_wells
    pct_med_int_le90 = 100.0 * med_int.dropna().le(90).sum() / n_wells

    return pd.DataFrame([
        {"statistic": "total_wells", "value": n_wells},
        {"statistic": "median_n_obs", "value": median_n_obs},
        {"statistic": "p75_n_obs", "value": p75_n_obs},
        {"statistic": "max_n_obs", "value": max_n_obs},
        {"statistic": "median_record_length_years", "value": median_record_years},
        {"statistic": "pct_wells_n_obs_le2", "value": round(pct_n_obs_le2, 2)},
        {"statistic": "pct_wells_median_interval_le30_days", "value": round(pct_med_int_le30, 2)},
        {"statistic": "pct_wells_median_interval_le90_days", "value": round(pct_med_int_le90, 2)},
    ])


def _plot_hist_median_sampling_interval(
    metrics: pd.DataFrame,
    output_path: str,
) -> None:
    """Histogram of median_sampling_interval_days with percentile annotations."""
    vals = metrics["median_sampling_interval_days"].dropna()
    if len(vals) == 0:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No valid median sampling intervals", ha="center", va="center")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    median_val = float(vals.median())
    p75 = float(vals.quantile(0.75))
    p90 = float(vals.quantile(0.9))

    fig, ax = plt.subplots(figsize=(8, 5))
    # Use fixed bins for reproducibility
    bins = np.arange(0, min(vals.max() + 50, 1000), 25)
    if len(bins) < 3:
        bins = np.linspace(0, vals.max() + 10, 25)
    ax.hist(vals, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(median_val, color="darkred", linestyle="--", linewidth=2, label=f"Median = {median_val:.0f} days")
    ax.axvline(p75, color="darkorange", linestyle="--", linewidth=1.5, label=f"75th %ile = {p75:.0f} days")
    ax.axvline(p90, color="green", linestyle="--", linewidth=1.5, label=f"90th %ile = {p90:.0f} days")
    ax.set_xlabel("Median sampling interval (days)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of median sampling interval per well")
    ax.legend(loc="upper right")
    txt = (
        f"Median = {median_val:.0f} days\n"
        f"75% < {p75:.0f} days\n"
        f"90% < {p90:.0f} days"
    )
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_hist_n_obs(metrics: pd.DataFrame, output_path: str) -> None:
    """Histogram of n_obs per well with log y-axis."""
    n_obs = metrics["n_obs"]
    median_val = float(n_obs.median())
    p75 = float(n_obs.quantile(0.75))
    max_val = int(n_obs.max())
    pct_le5 = 100.0 * (n_obs <= 5).sum() / len(metrics)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, min(max_val + 2, 150), 2)
    if len(bins) < 3:
        bins = np.linspace(0, max_val + 1, 30)
    counts, _, _ = ax.hist(n_obs, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Number of observations per well")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Distribution of observation count per well")
    txt = (
        f"Median n_obs = {median_val:.0f}\n"
        f"75% < {p75:.0f}\n"
        f"Max = {max_val}\n"
        f"{pct_le5:.1f}% of wells have ≤5 observations"
    )
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_scatter_record_length_vs_n_obs(metrics: pd.DataFrame, output_path: str) -> None:
    """Scatter: record_length_years vs n_obs with n_obs=10 reference line."""
    df = metrics[["record_length_years", "n_obs"]].dropna()
    pct_span_gt20 = 100.0 * (df["record_length_years"] > 20).sum() / len(metrics)
    pct_n_obs_ge10 = 100.0 * (df["n_obs"] >= 10).sum() / len(metrics)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["record_length_years"], df["n_obs"], alpha=0.4, s=15, c="steelblue", edgecolors="none")
    ax.axhline(10, color="darkred", linestyle="--", linewidth=2, label="n_obs = 10")
    ax.set_xlabel("Record length (years)")
    ax.set_ylabel("Number of observations")
    ax.set_title("Record length vs. observation count")
    ax.legend()
    txt = (
        f"{pct_span_gt20:.1f}% of wells span >20 years\n"
        f"but only {pct_n_obs_ge10:.1f}% have ≥10 observations"
    )
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_well_summary(
    df: pd.DataFrame,
    outdir: str,
    well_id_col: str = "well_id",
    date_col: str = "datetime",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full well data quality summary pipeline.

    Writes:
    - well_summary_metrics.csv
    - well_summary_overview.csv
    - hist_median_sampling_interval_days.png
    - hist_n_obs.png
    - scatter_record_length_vs_n_obs.png

    Returns
    -------
    tuple
        (metrics_df, overview_df)
    """
    os.makedirs(outdir, exist_ok=True)

    metrics = compute_well_summary_metrics(df, well_id_col=well_id_col, date_col=date_col)
    overview = compute_global_summary(metrics)

    metrics.to_csv(os.path.join(outdir, "well_summary_metrics.csv"), index=False)
    overview.to_csv(os.path.join(outdir, "well_summary_overview.csv"), index=False)

    _plot_hist_median_sampling_interval(
        metrics,
        os.path.join(outdir, "hist_median_sampling_interval_days.png"),
    )
    _plot_hist_n_obs(metrics, os.path.join(outdir, "hist_n_obs.png"))
    _plot_scatter_record_length_vs_n_obs(
        metrics,
        os.path.join(outdir, "scatter_record_length_vs_n_obs.png"),
    )

    return metrics, overview


def main_cli() -> None:
    """CLI entry point for well data quality summary."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize well data quality: n_obs, record length, sampling interval. No interpolation."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to CSV with well_id, datetime, wte")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory for CSVs and figures")
    parser.add_argument("--well-id-col", default="well_id", help="Column name for well ID")
    parser.add_argument("--date-col", default="datetime", help="Column name for date/datetime")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df[args.date_col] = pd.to_datetime(df[args.date_col])

    metrics, overview = run_well_summary(
        df,
        args.outdir,
        well_id_col=args.well_id_col,
        date_col=args.date_col,
    )
    print(f"Well summary complete. Outputs in {args.outdir}")
    print(f"  Wells: {len(metrics)}")
    print(overview.to_string(index=False))


if __name__ == "__main__":
    main_cli()
