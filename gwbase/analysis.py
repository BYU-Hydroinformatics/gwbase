"""
Statistical analysis functions for GWBASE.

This module implements advanced analysis methods:
- Mutual Information (MI) analysis
- Cross-Correlation Function (CCF) analysis
- Comparison of lag vs no-lag relationships
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy import signal
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from typing import Dict, Tuple


def calculate_mutual_info(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute mutual information between two continuous variables.

    Uses discretization to estimate MI from continuous data.

    Parameters
    ----------
    x : np.ndarray
        First variable (1D array)
    y : np.ndarray
        Second variable (1D array)
    n_bins : int, default 10
        Number of bins for discretization

    Returns
    -------
    float
        Mutual information score (bits)

    Example
    -------
    >>> mi = calculate_mutual_info(delta_wte, delta_q)
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 10:
        return np.nan

    # Discretize continuous variables
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    x_discrete = discretizer.fit_transform(x_clean.reshape(-1, 1)).flatten()
    y_discrete = discretizer.fit_transform(y_clean.reshape(-1, 1)).flatten()

    # Calculate mutual information
    mi = mutual_info_score(x_discrete, y_discrete)
    return mi


def calculate_well_metrics(
    well_data: pd.DataFrame,
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    n_bins: int = 10
) -> pd.Series:
    """
    Calculate MI, Pearson r, and Spearman r for a single well.

    Parameters
    ----------
    well_data : pd.DataFrame
        Data for a single well
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    n_bins : int, default 10
        Number of bins for MI calculation

    Returns
    -------
    pd.Series
        Metrics including MI, Pearson r, Spearman r

    Example
    -------
    >>> metrics = calculate_well_metrics(well_df)
    """
    delta_wte = well_data[delta_wte_col].values
    delta_q = well_data[delta_q_col].values

    # Mutual Information
    mi = calculate_mutual_info(delta_wte, delta_q, n_bins=n_bins)

    # Pearson Correlation
    if len(delta_wte) > 1 and np.std(delta_wte) > 0 and np.std(delta_q) > 0:
        pearson_corr, pearson_p = pearsonr(delta_wte, delta_q)
    else:
        pearson_corr, pearson_p = np.nan, np.nan

    # Spearman Correlation
    if len(delta_wte) > 1:
        spearman_corr, spearman_p = spearmanr(delta_wte, delta_q)
    else:
        spearman_corr, spearman_p = np.nan, np.nan

    return pd.Series({
        'mi': mi,
        'pearson_r': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p,
        'n_records': len(well_data)
    })


def compute_mi_analysis(
    data: pd.DataFrame,
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Compute mutual information analysis for all well-gage pairs.

    Parameters
    ----------
    data : pd.DataFrame
        Paired data with delta metrics
    well_id_col : str, default 'well_id'
        Column name for well ID
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    n_bins : int, default 10
        Number of bins for MI calculation

    Returns
    -------
    pd.DataFrame
        MI analysis results per well-gage pair

    Example
    -------
    >>> mi_results = compute_mi_analysis(data_with_deltas)
    """
    # Filter to valid data
    filtered = data.dropna(subset=[delta_wte_col, delta_q_col])

    # Compute metrics by well-gage pair
    results = (
        filtered
        .groupby([well_id_col, gage_id_col])
        .apply(lambda x: calculate_well_metrics(x, delta_wte_col, delta_q_col, n_bins))
        .reset_index()
    )

    print(f"MI Analysis Results:")
    print(f"  Well-gage pairs analyzed: {len(results)}")
    print(f"  Mean MI: {results['mi'].mean():.4f}")
    print(f"  Mean Pearson r: {results['pearson_r'].mean():.4f}")
    print(f"  Mean Spearman r: {results['spearman_r'].mean():.4f}")

    return results


def calculate_ccf(
    x: np.ndarray,
    y: np.ndarray,
    max_lag_days: int = 365
) -> Dict:
    """
    Calculate cross-correlation function between two time series.

    Parameters
    ----------
    x : np.ndarray
        First time series (e.g., delta_q)
    y : np.ndarray
        Second time series (e.g., delta_wte)
    max_lag_days : int, default 365
        Maximum lag to compute (days)

    Returns
    -------
    dict
        CCF results including lags, correlations, and optimal lag

    Example
    -------
    >>> ccf_result = calculate_ccf(delta_q, delta_wte)
    """
    # Standardize
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)

    # Calculate CCF
    correlation = signal.correlate(y, x, mode='full')
    lags = signal.correlation_lags(len(y), len(x), mode='full')

    # Limit to max_lag_days
    valid_indices = np.abs(lags) <= max_lag_days
    correlation = correlation[valid_indices]
    lags = lags[valid_indices]

    # Normalize
    correlation = correlation / len(x)

    # Find optimal lag
    optimal_idx = np.argmax(np.abs(correlation))

    return {
        'lags': lags,
        'correlation': correlation,
        'max_corr': np.max(np.abs(correlation)),
        'optimal_lag': lags[optimal_idx],
        'optimal_corr': correlation[optimal_idx]
    }


def compute_ccf_by_watershed(
    data: pd.DataFrame,
    gage_id_col: str = 'gage_id',
    well_id_col: str = 'well_id',
    delta_q_col: str = 'delta_q',
    delta_wte_col: str = 'delta_wte',
    date_col: str = 'date',
    max_lag_years: int = 10,
    min_data_years: float = 3.0
) -> Dict:
    """
    Calculate CCF between ΔQ and ΔWTE by watershed.

    Parameters
    ----------
    data : pd.DataFrame
        Paired data with delta metrics
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    well_id_col : str, default 'well_id'
        Column name for well ID
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE
    date_col : str, default 'date'
        Column name for date
    max_lag_years : int, default 10
        Maximum lag in years
    min_data_years : float, default 3.0
        Minimum years of data required

    Returns
    -------
    dict
        CCF results by watershed and well

    Example
    -------
    >>> ccf_results = compute_ccf_by_watershed(data_with_deltas)
    """
    max_lag_days = max_lag_years * 365
    min_data_points = int(min_data_years * 365)
    ccf_results = {}

    watersheds = data[gage_id_col].unique()
    print(f"Computing CCF for {len(watersheds)} watersheds...")

    for gage_id in watersheds:
        watershed_data = data[data[gage_id_col] == gage_id].copy()
        wells = watershed_data[well_id_col].unique()

        well_ccf_results = {}

        for well_id in wells:
            well_data = watershed_data[watershed_data[well_id_col] == well_id].copy()
            well_data = well_data.sort_values(date_col)

            # Check for sufficient data
            if len(well_data) < min_data_points:
                continue

            # Remove NaN values
            well_data = well_data.dropna(subset=[delta_q_col, delta_wte_col])
            if len(well_data) < min_data_points:
                continue

            x = well_data[delta_q_col].values
            y = well_data[delta_wte_col].values

            # Calculate CCF
            ccf_result = calculate_ccf(x, y, max_lag_days)
            ccf_result['n_points'] = len(well_data)
            ccf_result['data_years'] = len(well_data) / 365.25

            well_ccf_results[well_id] = ccf_result

        if well_ccf_results:
            ccf_results[gage_id] = well_ccf_results

    print(f"  Watersheds with CCF results: {len(ccf_results)}")

    return ccf_results


def compare_lag_vs_no_lag(
    no_lag_mi: pd.DataFrame,
    lag_mi: pd.DataFrame,
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Compare MI results between lagged and non-lagged analysis.

    Parameters
    ----------
    no_lag_mi : pd.DataFrame
        MI results without lag
    lag_mi : pd.DataFrame
        MI results with lag
    well_id_col : str, default 'well_id'
        Column name for well ID
    gage_id_col : str, default 'gage_id'
        Column name for gage ID

    Returns
    -------
    tuple
        (merged_results, by_gage_summary, overall_summary)

    Example
    -------
    >>> merged, by_gage, summary = compare_lag_vs_no_lag(no_lag_mi, lag_mi)
    """
    # Rename columns for clarity
    lag_mi = lag_mi.rename(columns={
        'mi': 'mi_lag',
        'pearson_r': 'pearson_lag',
        'spearman_r': 'spearman_lag',
        'n_records': 'n_records_lag'
    })
    no_lag_mi = no_lag_mi.rename(columns={
        'mi': 'mi_no_lag',
        'pearson_r': 'pearson_no_lag',
        'spearman_r': 'spearman_no_lag',
        'n_records': 'n_records_no_lag'
    })

    # Merge
    merged = pd.merge(
        no_lag_mi,
        lag_mi,
        on=[well_id_col, gage_id_col],
        how='inner'
    )

    # Calculate comparison metrics
    eps = 1e-9
    merged['delta_mi'] = merged['mi_lag'] - merged['mi_no_lag']
    merged['ratio_mi'] = merged['mi_lag'] / (merged['mi_no_lag'] + eps)

    # Non-linearity gain
    merged['nl_gain_lag'] = merged['mi_lag'] - merged['pearson_lag'].abs()
    merged['nl_gain_no_lag'] = merged['mi_no_lag'] - merged['pearson_no_lag'].abs()
    merged['delta_nl_gain'] = merged['nl_gain_lag'] - merged['nl_gain_no_lag']

    # Aggregate by gage
    by_gage = merged.groupby(gage_id_col).agg(
        n_wells=(well_id_col, 'count'),
        mi_lag_mean=('mi_lag', 'mean'),
        mi_no_lag_mean=('mi_no_lag', 'mean'),
        delta_mi_mean=('delta_mi', 'mean'),
        pct_wells_lag_higher=('delta_mi', lambda x: (x > 0).mean() * 100)
    ).reset_index().sort_values('delta_mi_mean', ascending=False)

    # Overall summary
    summary = {
        'n_pairs': len(merged),
        'mi_lag_mean': merged['mi_lag'].mean(),
        'mi_no_lag_mean': merged['mi_no_lag'].mean(),
        'delta_mi_mean': merged['delta_mi'].mean(),
        'delta_mi_median': merged['delta_mi'].median(),
        'pct_pairs_lag_higher': (merged['delta_mi'] > 0).mean() * 100
    }

    print("\nLag vs No-Lag Comparison:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return merged, by_gage, summary


def aggregate_ccf_results(ccf_results: Dict) -> pd.DataFrame:
    """
    Aggregate CCF results into a summary DataFrame.

    Parameters
    ----------
    ccf_results : dict
        CCF results from compute_ccf_by_watershed

    Returns
    -------
    pd.DataFrame
        Summary of optimal lags and correlations

    Example
    -------
    >>> ccf_summary = aggregate_ccf_results(ccf_results)
    """
    records = []

    for gage_id, wells in ccf_results.items():
        for well_id, result in wells.items():
            records.append({
                'gage_id': gage_id,
                'well_id': well_id,
                'optimal_lag_days': result['optimal_lag'],
                'optimal_lag_years': result['optimal_lag'] / 365.25,
                'max_correlation': result['max_corr'],
                'optimal_correlation': result['optimal_corr'],
                'n_points': result['n_points'],
                'data_years': result['data_years']
            })

    df = pd.DataFrame(records)

    print(f"\nCCF Summary:")
    print(f"  Well-gage pairs: {len(df)}")
    print(f"  Mean optimal lag: {df['optimal_lag_days'].mean():.1f} days ({df['optimal_lag_years'].mean():.2f} years)")
    print(f"  Median optimal lag: {df['optimal_lag_days'].median():.1f} days")
    print(f"  Mean max correlation: {df['max_correlation'].mean():.4f}")

    return df
