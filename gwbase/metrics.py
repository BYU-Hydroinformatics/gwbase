"""
Delta metrics computation for GWBASE.

This module implements Steps 8-9 of the GWBASE workflow:
- Compute ΔWTE (change in water table elevation)
- Compute ΔQ (change in streamflow)
- Analyze ΔWTE–ΔQ relationships via linear regression
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr
from typing import Tuple, Dict, List, Optional


def compute_delta_metrics(
    paired_data: pd.DataFrame,
    well_id_col: str = 'well_id',
    wte_col: str = 'wte',
    q_col: str = 'q',
    wte0_col: str = 'wte0',
    q0_col: str = 'q0'
) -> pd.DataFrame:
    """
    Compute ΔWTE and ΔQ from baseline values.

    ΔWTE = WTE - WTE0 (change in water table elevation)
    ΔQ = Q - Q0 (change in streamflow)

    Parameters
    ----------
    paired_data : pd.DataFrame
        Paired data with baseline values (wte0, q0)
    well_id_col : str, default 'well_id'
        Column name for well ID
    wte_col : str, default 'wte'
        Column name for water table elevation
    q_col : str, default 'q'
        Column name for streamflow
    wte0_col : str, default 'wte0'
        Column name for baseline WTE
    q0_col : str, default 'q0'
        Column name for baseline Q

    Returns
    -------
    pd.DataFrame
        Data with delta_wte and delta_q columns added

    Example
    -------
    >>> data_with_deltas = compute_delta_metrics(paired_data)
    """
    data = paired_data.copy()

    # Compute delta values
    data['delta_wte'] = data[wte_col] - data[wte0_col]
    data['delta_q'] = data[q_col] - data[q0_col]

    # Summary statistics
    valid_delta = data['delta_wte'].notna() & data['delta_q'].notna()

    print(f"Delta metrics computed:")
    print(f"  Valid delta records: {valid_delta.sum():,}")
    print(f"  ΔWTE range: {data['delta_wte'].min():.2f} to {data['delta_wte'].max():.2f} ft")
    print(f"  ΔQ range: {data['delta_q'].min():.2f} to {data['delta_q'].max():.2f} cfs")

    return data


def create_lag_analysis(
    data: pd.DataFrame,
    lag_period: int,
    period_unit: str = 'years',
    well_id_col: str = 'well_id',
    date_col: str = 'date',
    delta_wte_col: str = 'delta_wte'
) -> pd.DataFrame:
    """
    Create lagged ΔWTE values for lag analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Data with delta_wte column
    lag_period : int
        Number of time units to lag
    period_unit : str, default 'years'
        Time unit ('years', 'months', 'days')
    well_id_col : str, default 'well_id'
        Column name for well ID
    date_col : str, default 'date'
        Column name for date
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE

    Returns
    -------
    pd.DataFrame
        Data with lagged ΔWTE column

    Example
    -------
    >>> lag_1yr = create_lag_analysis(data, 1, 'years')
    """
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    # Create lag date column based on period_unit
    if period_unit == 'years':
        lag_date_col = f'date_{lag_period}_year'
        data[lag_date_col] = data[date_col] - pd.DateOffset(years=lag_period)
        lag_col_name = f'delta_wte_lag_{lag_period}_year{"s" if lag_period > 1 else ""}'
    elif period_unit == 'months':
        lag_date_col = f'date_{lag_period}_month'
        data[lag_date_col] = data[date_col] - pd.DateOffset(months=lag_period)
        lag_col_name = f'delta_wte_lag_{lag_period}_month{"s" if lag_period > 1 else ""}'
    elif period_unit == 'days':
        lag_date_col = f'date_{lag_period}_day'
        data[lag_date_col] = data[date_col] - pd.DateOffset(days=lag_period)
        lag_col_name = f'delta_wte_lag_{lag_period}_day{"s" if lag_period > 1 else ""}'
    else:
        raise ValueError("period_unit must be 'years', 'months', or 'days'")

    # Create lookup table for lag values
    lookup = data[[well_id_col, date_col, delta_wte_col]].rename(
        columns={date_col: lag_date_col, delta_wte_col: lag_col_name}
    )

    # Merge to get lagged values
    lag_analysis = data.merge(lookup, on=[well_id_col, lag_date_col], how='inner')

    print(f"Lag analysis ({lag_period} {period_unit}):")
    print(f"  Input records: {len(data):,}")
    print(f"  Records with lag data: {len(lag_analysis):,}")

    return lag_analysis


def compute_regression_by_gage(
    data: pd.DataFrame,
    gage_id_col: str = 'gage_id',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    min_observations: int = 10
) -> pd.DataFrame:
    """
    Compute linear regression statistics for ΔWTE vs ΔQ by gage.

    Parameters
    ----------
    data : pd.DataFrame
        Data with delta_wte and delta_q columns
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    min_observations : int, default 10
        Minimum observations required for regression

    Returns
    -------
    pd.DataFrame
        Regression statistics per gage

    Example
    -------
    >>> gage_stats = compute_regression_by_gage(data_with_deltas)
    """
    results = []

    for gage_id, group in data.groupby(gage_id_col):
        # Clean data
        clean = group.dropna(subset=[delta_wte_col, delta_q_col])

        if len(clean) < min_observations:
            continue

        x = clean[delta_wte_col].values
        y = clean[delta_q_col].values

        # Check for variance
        if np.std(x) == 0 or np.std(y) == 0:
            continue

        # Compute regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        results.append({
            'gage_id': gage_id,
            'n_wells': clean['well_id'].nunique() if 'well_id' in clean.columns else None,
            'n_observations': len(clean),
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err
        })

    results_df = pd.DataFrame(results)

    print(f"Regression analysis by gage:")
    print(f"  Gages analyzed: {len(results_df)}")
    print(f"  Mean R²: {results_df['r_squared'].mean():.4f}")
    print(f"  Median R²: {results_df['r_squared'].median():.4f}")
    print(f"  Significant (p<0.05): {(results_df['p_value'] < 0.05).sum()}")

    return results_df


def compute_regression_by_well(
    data: pd.DataFrame,
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    min_observations: int = 10
) -> pd.DataFrame:
    """
    Compute linear regression statistics for ΔWTE vs ΔQ by well.

    Parameters
    ----------
    data : pd.DataFrame
        Data with delta_wte and delta_q columns
    well_id_col : str, default 'well_id'
        Column name for well ID
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    min_observations : int, default 10
        Minimum observations required for regression

    Returns
    -------
    pd.DataFrame
        Regression statistics per well

    Example
    -------
    >>> well_stats = compute_regression_by_well(data_with_deltas)
    """
    results = []

    for (well_id, gage_id), group in data.groupby([well_id_col, gage_id_col]):
        # Clean data
        clean = group.dropna(subset=[delta_wte_col, delta_q_col])

        if len(clean) < min_observations:
            continue

        x = clean[delta_wte_col].values
        y = clean[delta_q_col].values

        # Check for variance
        if np.std(x) == 0 or np.std(y) == 0:
            continue

        # Compute regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Compute correlations
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)

        results.append({
            'well_id': well_id,
            'gage_id': gage_id,
            'n_observations': len(clean),
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        })

    results_df = pd.DataFrame(results)

    print(f"Regression analysis by well:")
    print(f"  Wells analyzed: {len(results_df)}")
    print(f"  Mean R²: {results_df['r_squared'].mean():.4f}")
    print(f"  Median R²: {results_df['r_squared'].median():.4f}")

    return results_df


def filter_by_correlation(
    data: pd.DataFrame,
    well_stats: pd.DataFrame,
    percentile: float = 10,
    correlation_col: str = 'r_squared',
    well_id_col: str = 'well_id'
) -> pd.DataFrame:
    """
    Filter data to wells in the top percentile of correlation.

    Parameters
    ----------
    data : pd.DataFrame
        Full paired data
    well_stats : pd.DataFrame
        Regression statistics per well
    percentile : float, default 10
        Top percentile to keep (e.g., 10 = top 10%)
    correlation_col : str, default 'r_squared'
        Column to use for ranking
    well_id_col : str, default 'well_id'
        Column name for well ID

    Returns
    -------
    pd.DataFrame
        Data filtered to top correlation wells

    Example
    -------
    >>> top_wells_data = filter_by_correlation(data, well_stats, percentile=10)
    """
    # Calculate threshold
    threshold = np.percentile(well_stats[correlation_col], 100 - percentile)

    # Get top wells
    top_wells = well_stats[well_stats[correlation_col] >= threshold][well_id_col].values

    # Filter data
    filtered = data[data[well_id_col].isin(top_wells)].copy()

    print(f"Correlation filter (top {percentile}%):")
    print(f"  Threshold {correlation_col}: {threshold:.4f}")
    print(f"  Wells retained: {len(top_wells)}")
    print(f"  Records retained: {len(filtered):,}")

    return filtered


def summarize_regression_results(
    gage_stats: pd.DataFrame,
    well_stats: pd.DataFrame = None
) -> Dict:
    """
    Generate summary statistics from regression analysis.

    Parameters
    ----------
    gage_stats : pd.DataFrame
        Regression statistics by gage
    well_stats : pd.DataFrame, optional
        Regression statistics by well

    Returns
    -------
    dict
        Summary statistics

    Example
    -------
    >>> summary = summarize_regression_results(gage_stats, well_stats)
    """
    summary = {
        'n_gages': len(gage_stats),
        'gage_r2_mean': gage_stats['r_squared'].mean(),
        'gage_r2_median': gage_stats['r_squared'].median(),
        'gage_r2_std': gage_stats['r_squared'].std(),
        'gage_slope_mean': gage_stats['slope'].mean(),
        'gage_slope_median': gage_stats['slope'].median(),
        'gage_significant_count': (gage_stats['p_value'] < 0.05).sum(),
        'gage_significant_pct': (gage_stats['p_value'] < 0.05).mean() * 100,
        'gage_positive_slope_count': (gage_stats['slope'] > 0).sum(),
        'gage_positive_slope_pct': (gage_stats['slope'] > 0).mean() * 100
    }

    if well_stats is not None:
        summary.update({
            'n_wells': len(well_stats),
            'well_r2_mean': well_stats['r_squared'].mean(),
            'well_r2_median': well_stats['r_squared'].median(),
            'well_r2_std': well_stats['r_squared'].std(),
            'well_slope_mean': well_stats['slope'].mean(),
            'well_slope_median': well_stats['slope'].median(),
            'well_significant_count': (well_stats['p_value'] < 0.05).sum(),
            'well_significant_pct': (well_stats['p_value'] < 0.05).mean() * 100
        })

    print("\nRegression Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return summary
