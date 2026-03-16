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
import pymannkendall as mk
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
    # Ensure numeric dtype and coerce non-numeric values to NaN
    x = pd.to_numeric(pd.Series(x), errors='coerce').to_numpy()
    y = pd.to_numeric(pd.Series(y), errors='coerce').to_numpy()

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
    )
    
    # Reset index - handle case where gage_id might already be in index
    if isinstance(results.index, pd.MultiIndex):
        results = results.reset_index()
    else:
        results = results.reset_index()
        # If gage_id is already a column, drop the duplicate
        if 'gage_id' in results.columns and results.index.name == 'gage_id':
            results = results.drop(columns=['gage_id'], errors='ignore')

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

    # Normalize gage_id types to string before merging
    no_lag_mi = no_lag_mi.copy()
    lag_mi = lag_mi.copy()
    no_lag_mi[gage_id_col] = no_lag_mi[gage_id_col].astype(str)
    lag_mi[gage_id_col] = lag_mi[gage_id_col].astype(str)

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


def compute_seasonal_monthly_analysis(
    data: pd.DataFrame,
    date_col: str = 'date',
    gage_id_col: str = 'gage_id',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q',
    min_observations: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute regression statistics (ΔQ vs ΔWTE) by season and by month.

    Seasons: Winter (Dec-Feb), Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Nov)

    Parameters
    ----------
    data : pd.DataFrame
        Data with delta_wte, delta_q, date, gage_id
    date_col : str, default 'date'
        Column name for date
    gage_id_col : str, default 'gage_id'
        Column name for gage ID
    delta_wte_col : str, default 'delta_wte'
        Column name for ΔWTE
    delta_q_col : str, default 'delta_q'
        Column name for ΔQ
    min_observations : int, default 5
        Minimum observations for regression

    Returns
    -------
    tuple
        (seasonal_stats, monthly_stats)
    """
    from scipy.stats import linregress

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[delta_wte_col, delta_q_col, date_col, gage_id_col])

    def _get_season(month: int) -> str:
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    df['month'] = df[date_col].dt.month
    df['season'] = df['month'].apply(_get_season)

    def _regress_group(g):
        if len(g) < min_observations:
            return None
        x, y = g[delta_wte_col].values, g[delta_q_col].values
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        try:
            slope, intercept, r, p, se = linregress(x, y)
            return {'slope': slope, 'r_squared': r ** 2, 'p_value': p, 'n_obs': len(g)}
        except (ValueError, AttributeError):
            return None

    # Seasonal stats
    seasonal_records = []
    for (gage_id, season), grp in df.groupby([gage_id_col, 'season']):
        res = _regress_group(grp)
        if res:
            seasonal_records.append({
                gage_id_col: gage_id,
                'season': season,
                'n_observations': res['n_obs'],
                'slope': res['slope'],
                'r_squared': res['r_squared'],
                'p_value': res['p_value']
            })
    seasonal_stats = pd.DataFrame(seasonal_records)

    # Monthly stats
    monthly_records = []
    for (gage_id, month), grp in df.groupby([gage_id_col, 'month']):
        res = _regress_group(grp)
        if res:
            monthly_records.append({
                gage_id_col: gage_id,
                'month': month,
                'month_name': month_names[month],
                'n_observations': res['n_obs'],
                'slope': res['slope'],
                'r_squared': res['r_squared'],
                'p_value': res['p_value']
            })
    monthly_stats = pd.DataFrame(monthly_records)

    print(f"\nSeasonal/Monthly Analysis:")
    print(f"  Seasonal: {len(seasonal_stats)} gage-season combinations")
    print(f"  Monthly: {len(monthly_stats)} gage-month combinations")
    if len(seasonal_stats) > 0:
        print(f"  Seasonal mean R²: {seasonal_stats['r_squared'].mean():.4f}")
    if len(monthly_stats) > 0:
        print(f"  Monthly mean R²: {monthly_stats['r_squared'].mean():.4f}")

    return seasonal_stats, monthly_stats


def combine_regression_summary(
    gage_stats: pd.DataFrame,
    seasonal_stats: pd.DataFrame,
    monthly_stats: pd.DataFrame,
    gage_id_col: str = 'gage_id'
) -> pd.DataFrame:
    """
    Combine overall, seasonal, and monthly regression results (R², slope) into one table.

    Returns
    -------
    pd.DataFrame
        Combined table with columns: gage_id, period_type, period, n_observations, slope, r_squared, p_value
    """
    records = []

    # Overall
    for _, row in gage_stats.iterrows():
        records.append({
            gage_id_col: row[gage_id_col],
            'period_type': 'overall',
            'period': 'overall',
            'n_observations': row.get('n_observations', np.nan),
            'slope': row.get('slope', np.nan),
            'r_squared': row.get('r_squared', np.nan),
            'p_value': row.get('p_value', np.nan)
        })

    # Seasonal
    for _, row in seasonal_stats.iterrows():
        records.append({
            gage_id_col: row[gage_id_col],
            'period_type': 'seasonal',
            'period': row['season'],
            'n_observations': row.get('n_observations', np.nan),
            'slope': row.get('slope', np.nan),
            'r_squared': row.get('r_squared', np.nan),
            'p_value': row.get('p_value', np.nan)
        })

    # Monthly
    for _, row in monthly_stats.iterrows():
        records.append({
            gage_id_col: row[gage_id_col],
            'period_type': 'monthly',
            'period': row.get('month_name', row.get('month', '')),
            'n_observations': row.get('n_observations', np.nan),
            'slope': row.get('slope', np.nan),
            'r_squared': row.get('r_squared', np.nan),
            'p_value': row.get('p_value', np.nan)
        })

    combined = pd.DataFrame(records)

    # Order: gage, then overall -> seasonal -> monthly, then by period
    type_order = {'overall': 0, 'seasonal': 1, 'monthly': 2}
    period_order = ['overall', 'Winter', 'Spring', 'Summer', 'Fall',
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    combined['_type_key'] = combined['period_type'].map(type_order)
    combined['_period_key'] = combined['period'].map(
        lambda p: period_order.index(p) if p in period_order else 999
    )
    combined = combined.sort_values([gage_id_col, '_type_key', '_period_key'])
    combined = combined.drop(columns=['_type_key', '_period_key'])

    return combined


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


# ── Mann-Kendall + Sen's Slope ────────────────────────────────────────────────

def _mk_row(series: pd.Series, dates: pd.Series) -> dict:
    """Run MK test on a time series; return a result dict."""
    x = series.dropna().values
    if len(x) < 4:
        return {
            'trend': 'insufficient data', 'h': False,
            'p_value': np.nan, 'z': np.nan, 'tau': np.nan,
            'sen_slope': np.nan, 'intercept': np.nan,
        }
    try:
        res = mk.original_test(x)
        return {
            'trend': res.trend,
            'h': bool(res.h),
            'p_value': float(res.p),
            'z': float(res.z),
            'tau': float(res.Tau),
            'sen_slope': float(res.slope),   # units per time step
            'intercept': float(res.intercept),
        }
    except Exception:
        return {
            'trend': 'error', 'h': False,
            'p_value': np.nan, 'z': np.nan, 'tau': np.nan,
            'sen_slope': np.nan, 'intercept': np.nan,
        }


def compute_mk_well_wte(
    well_ts: pd.DataFrame,
    well_id_col: str = 'well_id',
    date_col: str = 'date',
    wte_col: str = 'WTE',
    min_obs: int = 10,
) -> pd.DataFrame:
    """
    Mann-Kendall test and Sen's slope on WTE time series for each well.

    Sen's slope unit: feet per year (converted from ft per observation using
    median inter-observation interval).

    Parameters
    ----------
    well_ts : pd.DataFrame
        Raw or cleaned well time series (one row per measurement).
    min_obs : int
        Minimum observations required to run the test.

    Returns
    -------
    pd.DataFrame
        One row per well with MK statistics and Sen's slope (ft/yr).
    """
    well_ts = well_ts.copy()
    well_ts[date_col] = pd.to_datetime(well_ts[date_col])
    well_ts = well_ts.sort_values([well_id_col, date_col])

    results = []
    for wid, grp in well_ts.groupby(well_id_col):
        grp = grp.dropna(subset=[wte_col]).sort_values(date_col)
        n = len(grp)
        if n < min_obs:
            continue

        row = _mk_row(grp[wte_col], grp[date_col])

        # Convert Sen's slope from ft/step to ft/yr
        if n >= 2:
            intervals = grp[date_col].diff().dt.days.dropna()
            median_interval_days = intervals.median()
            if median_interval_days > 0:
                row['sen_slope_yr'] = row['sen_slope'] * (365.25 / median_interval_days)
            else:
                row['sen_slope_yr'] = np.nan
        else:
            row['sen_slope_yr'] = np.nan

        yr_span = (grp[date_col].max() - grp[date_col].min()).days / 365.25
        row.update({
            well_id_col: wid,
            'n_obs': n,
            'year_span': round(yr_span, 2),
            'date_start': grp[date_col].min().date(),
            'date_end': grp[date_col].max().date(),
        })
        results.append(row)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Reorder columns
    front = [well_id_col, 'n_obs', 'year_span', 'date_start', 'date_end',
             'trend', 'h', 'p_value', 'z', 'tau', 'sen_slope_yr', 'sen_slope', 'intercept']
    df = df[[c for c in front if c in df.columns]]

    sig = (df['p_value'] < 0.05).sum()
    dec = ((df['p_value'] < 0.05) & (df['sen_slope_yr'] < 0)).sum()
    inc = ((df['p_value'] < 0.05) & (df['sen_slope_yr'] > 0)).sum()
    print(f"MK test - well WTE:")
    print(f"  Wells tested: {len(df)}")
    print(f"  Significant (p<0.05): {sig} ({sig/len(df)*100:.1f}%)")
    print(f"    Declining: {dec}  |  Increasing: {inc}")
    print(f"  Median Sen's slope: {df['sen_slope_yr'].median():.4f} ft/yr")
    return df


def compute_mk_streamflow(
    streamflow_monthly: pd.DataFrame,
    gage_id_col: str = 'gage_id',
    date_col: str = 'date',
    q_col: str = 'q',
    min_obs: int = 10,
) -> pd.DataFrame:
    """
    Mann-Kendall test and Sen's slope on monthly BFD=1 streamflow for each gage.

    Sen's slope unit: cfs per year.

    Parameters
    ----------
    streamflow_monthly : pd.DataFrame
        Monthly aggregated streamflow (one row per gage-month).

    Returns
    -------
    pd.DataFrame
        One row per gage with MK statistics and Sen's slope (cfs/yr).
    """
    sf = streamflow_monthly.copy()
    sf[date_col] = pd.to_datetime(sf[date_col])
    sf = sf.sort_values([gage_id_col, date_col])

    results = []
    for gid, grp in sf.groupby(gage_id_col):
        grp = grp.dropna(subset=[q_col]).sort_values(date_col)
        n = len(grp)
        if n < min_obs:
            continue

        row = _mk_row(grp[q_col], grp[date_col])
        # Monthly data: slope per month -> per year
        row['sen_slope_yr'] = row['sen_slope'] * 12

        yr_span = (grp[date_col].max() - grp[date_col].min()).days / 365.25
        row.update({
            gage_id_col: gid,
            'n_obs': n,
            'year_span': round(yr_span, 2),
            'date_start': grp[date_col].min().date(),
            'date_end': grp[date_col].max().date(),
        })
        results.append(row)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    front = [gage_id_col, 'n_obs', 'year_span', 'date_start', 'date_end',
             'trend', 'h', 'p_value', 'z', 'tau', 'sen_slope_yr', 'sen_slope', 'intercept']
    df = df[[c for c in front if c in df.columns]]

    sig = (df['p_value'] < 0.05).sum()
    print(f"MK test - gage streamflow:")
    print(f"  Gages tested: {len(df)}")
    print(f"  Significant (p<0.05): {sig}")
    print(f"  Median Sen's slope: {df['sen_slope_yr'].median():.4f} cfs/yr")
    return df


def compute_mk_gage_wte(
    data_with_deltas: pd.DataFrame,
    gage_id_col: str = 'gage_id',
    date_col: str = 'date',
    wte_col: str = 'wte',
    min_obs: int = 10,
) -> pd.DataFrame:
    """
    Mann-Kendall test and Sen's slope on per-gage mean monthly WTE
    (aggregated across all wells in the catchment).

    Sen's slope unit: feet per year.

    Parameters
    ----------
    data_with_deltas : pd.DataFrame
        Paired delta data with well_id, gage_id, date, wte columns.

    Returns
    -------
    pd.DataFrame
        One row per gage.
    """
    df = data_with_deltas.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Aggregate: mean WTE across all wells per gage per month
    monthly_mean = (
        df.groupby([gage_id_col, date_col])[wte_col]
        .mean()
        .reset_index()
        .sort_values([gage_id_col, date_col])
    )

    results = []
    for gid, grp in monthly_mean.groupby(gage_id_col):
        grp = grp.dropna(subset=[wte_col]).sort_values(date_col)
        n = len(grp)
        if n < min_obs:
            continue

        row = _mk_row(grp[wte_col], grp[date_col])
        row['sen_slope_yr'] = row['sen_slope'] * 12  # monthly -> annual

        yr_span = (grp[date_col].max() - grp[date_col].min()).days / 365.25
        n_wells = df[df[gage_id_col] == gid]['well_id'].nunique() if 'well_id' in df.columns else np.nan
        row.update({
            gage_id_col: gid,
            'n_months': n,
            'n_wells': int(n_wells) if (not isinstance(n_wells, float) or not np.isnan(n_wells)) else np.nan,
            'year_span': round(yr_span, 2),
            'date_start': grp[date_col].min().date(),
            'date_end': grp[date_col].max().date(),
        })
        results.append(row)

    df_out = pd.DataFrame(results)
    if df_out.empty:
        return df_out

    front = [gage_id_col, 'n_months', 'n_wells', 'year_span', 'date_start', 'date_end',
             'trend', 'h', 'p_value', 'z', 'tau', 'sen_slope_yr', 'sen_slope', 'intercept']
    df_out = df_out[[c for c in front if c in df_out.columns]]

    sig = (df_out['p_value'] < 0.05).sum()
    print(f"MK test - gage mean WTE (aggregated):")
    print(f"  Gages tested: {len(df_out)}")
    print(f"  Significant (p<0.05): {sig}")
    print(f"  Median Sen's slope: {df_out['sen_slope_yr'].median():.4f} ft/yr")
    return df_out
