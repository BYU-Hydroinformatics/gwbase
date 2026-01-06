"""Data quality control and filtering."""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def filter_wells_by_quality(
    well_ts: pd.DataFrame,
    well_id_col: str = 'well_id',
    date_col: str = 'date',
    wte_col: str = 'wte',
    min_observations: int = 5,
    min_time_span_days: int = 730,
    z_score_threshold: float = 3.0,
    iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Filter wells based on data quality criteria and remove outliers.
    
    Parameters:
    -----------
    well_ts : DataFrame
        Well time series data
    well_id_col : str
        Column name for well ID
    date_col : str
        Column name for date
    wte_col : str
        Column name for water table elevation
    min_observations : int
        Minimum number of observations required per well
    min_time_span_days : int
        Minimum time span in days required per well
    z_score_threshold : float
        Z-score threshold for outlier detection
    iqr_multiplier : float
        IQR multiplier for outlier detection
        
    Returns:
    --------
    DataFrame
        Filtered well data with outliers removed
    """
    logger.info(f"Filtering wells by quality criteria")
    
    well_ts = well_ts.copy()
    well_ts[date_col] = pd.to_datetime(well_ts[date_col])
    
    # Filter by observation count and time span
    initial_wells = well_ts[well_id_col].nunique()
    well_stats = well_ts.groupby(well_id_col).agg({
        date_col: ['count', 'min', 'max'],
        wte_col: 'count'
    }).reset_index()
    
    well_stats.columns = [well_id_col, 'n_obs', 'start_date', 'end_date', 'wte_count']
    well_stats['time_span_days'] = (
        pd.to_datetime(well_stats['end_date']) - pd.to_datetime(well_stats['start_date'])
    ).dt.days
    
    # Filter wells
    valid_wells = well_stats[
        (well_stats['n_obs'] >= min_observations) &
        (well_stats['time_span_days'] >= min_time_span_days)
    ][well_id_col].unique()
    
    well_ts_filtered = well_ts[well_ts[well_id_col].isin(valid_wells)].copy()
    logger.info(f"Filtered from {initial_wells} to {len(valid_wells)} wells")
    
    # Remove outliers using Z-score and IQR methods
    outlier_results = []
    for well_id, group in well_ts_filtered.groupby(well_id_col):
        if len(group) < min_observations:
            continue
        
        wte_values = group[wte_col].values
        is_outlier = np.zeros(len(group), dtype=bool)
        
        # Z-score method
        try:
            z_scores = np.abs(stats.zscore(wte_values, nan_policy='omit'))
            is_zscore_outlier = z_scores > z_score_threshold
        except:
            is_zscore_outlier = np.zeros(len(group), dtype=bool)
        
        # IQR method
        try:
            Q1, Q3 = np.nanpercentile(wte_values, [25, 75])
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                is_iqr_outlier = (wte_values < lower_bound) | (wte_values > upper_bound)
            else:
                is_iqr_outlier = np.zeros(len(group), dtype=bool)
        except:
            is_iqr_outlier = np.zeros(len(group), dtype=bool)
        
        # Combine methods
        is_outlier = is_zscore_outlier | is_iqr_outlier
        
        group_outliers = group.copy()
        group_outliers['is_outlier'] = is_outlier
        outlier_results.append(group_outliers)
    
    if not outlier_results:
        return pd.DataFrame(columns=well_ts.columns)
    
    result_with_outliers = pd.concat(outlier_results, ignore_index=True)
    result_clean = result_with_outliers[~result_with_outliers['is_outlier']].copy()
    result_clean = result_clean.drop(columns=['is_outlier'])
    
    logger.info(f"Removed {len(result_with_outliers) - len(result_clean):,} outlier observations")
    return result_clean

