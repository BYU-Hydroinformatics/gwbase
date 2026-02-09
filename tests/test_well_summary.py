"""Unit tests for well_summary module."""

import os
import tempfile

import pandas as pd
import pytest

from gwbase.well_summary import (
    compute_well_summary_metrics,
    compute_global_summary,
    run_well_summary,
)


def test_compute_well_summary_metrics_single_obs():
    """Single observation: n_obs=1, record_length=0, median_interval=NaN."""
    df = pd.DataFrame({
        "well_id": ["w1"],
        "datetime": ["2020-01-15"],
        "wte": [100.0],
    })
    m = compute_well_summary_metrics(df, well_id_col="well_id", date_col="datetime")
    assert len(m) == 1
    assert m["n_obs"].iloc[0] == 1
    assert m["record_length_years"].iloc[0] == 0.0
    assert pd.isna(m["median_sampling_interval_days"].iloc[0])


def test_compute_well_summary_metrics_two_obs():
    """Two observations: record_length and median_interval computed."""
    df = pd.DataFrame({
        "well_id": ["w1", "w1"],
        "datetime": ["2020-01-01", "2020-02-01"],
        "wte": [100.0, 101.0],
    })
    m = compute_well_summary_metrics(df, well_id_col="well_id", date_col="datetime")
    assert len(m) == 1
    assert m["n_obs"].iloc[0] == 2
    assert 0.08 < m["record_length_years"].iloc[0] < 0.09  # ~31 days
    assert 30 < m["median_sampling_interval_days"].iloc[0] < 32


def test_compute_well_summary_metrics_multiple_wells():
    """Multiple wells with varying observation counts."""
    df = pd.DataFrame({
        "well_id": ["w1", "w1", "w1", "w2", "w2"],
        "datetime": ["2020-01-01", "2020-02-01", "2020-03-01", "2019-06-15", "2021-06-15"],
        "wte": [100.0, 101.0, 102.0, 95.0, 96.0],
    })
    m = compute_well_summary_metrics(df, well_id_col="well_id", date_col="datetime")
    assert len(m) == 2
    w1 = m[m["well_id"] == "w1"].iloc[0]
    assert w1["n_obs"] == 3
    assert w1["median_sampling_interval_days"] == pytest.approx(29.5, rel=1)
    w2 = m[m["well_id"] == "w2"].iloc[0]
    assert w2["n_obs"] == 2
    assert 700 < w2["median_sampling_interval_days"] < 735  # ~2 years


def test_compute_global_summary():
    """Global summary produces expected columns."""
    m = pd.DataFrame({
        "well_id": ["w1", "w2", "w3"],
        "n_obs": [1, 5, 100],
        "record_length_years": [0.0, 2.0, 20.0],
        "median_sampling_interval_days": [float("nan"), 30.0, 60.0],
    })
    s = compute_global_summary(m)
    assert "total_wells" in s["statistic"].values
    assert s[s["statistic"] == "total_wells"]["value"].iloc[0] == 3
    assert s[s["statistic"] == "median_n_obs"]["value"].iloc[0] == 5
    assert s[s["statistic"] == "max_n_obs"]["value"].iloc[0] == 100


def test_run_well_summary():
    """Full pipeline produces CSVs and figures."""
    df = pd.DataFrame({
        "well_id": ["w1", "w1", "w1", "w2", "w2"],
        "datetime": ["2020-01-01", "2020-02-01", "2020-03-01", "2019-01-01", "2021-01-01"],
        "wte": [100.0, 101.0, 102.0, 95.0, 96.0],
    })
    with tempfile.TemporaryDirectory() as outdir:
        metrics, overview = run_well_summary(
            df, outdir, well_id_col="well_id", date_col="datetime"
        )
        assert len(metrics) == 2
        assert os.path.exists(os.path.join(outdir, "well_summary_metrics.csv"))
        assert os.path.exists(os.path.join(outdir, "well_summary_overview.csv"))
        assert os.path.exists(os.path.join(outdir, "hist_median_sampling_interval_days.png"))
        assert os.path.exists(os.path.join(outdir, "hist_n_obs.png"))
        assert os.path.exists(os.path.join(outdir, "scatter_record_length_vs_n_obs.png"))
