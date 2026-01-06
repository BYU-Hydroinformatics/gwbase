"""PCHIP interpolation for groundwater time series."""
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def pchip_daily_interpolation(
    well_ts: pd.DataFrame,
    well_id_col: str = 'well_id',
    date_col: str = 'date',
    wte_col: str = 'wte',
    target_frequency: str = 'D',
    extrapolate: bool = False,
    max_gap_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Perform daily PCHIP interpolation on groundwater well time series data.
    
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) preserves
    monotonicity and avoids unrealistic oscillations in hydrologic data.
    
    Based on notebook: 03_pchip_interpolation.ipynb, cells 17-18
    
    Parameters:
    -----------
    well_ts : DataFrame
        Well time series with columns: well_id, date, wte
    well_id_col : str
        Column name for well ID
    date_col : str
        Column name for date
    wte_col : str
        Column name for water table elevation
    target_frequency : str
        Target frequency ('D' for daily)
    extrapolate : bool
        Allow extrapolation beyond data range (not recommended)
    max_gap_days : int, optional
        Maximum gap size to interpolate (None = no limit)
        
    Returns:
    --------
    DataFrame
        Daily interpolated values with columns: well_id, date, wte
    """
    logger.info(f"Starting PCHIP interpolation for {well_ts[well_id_col].nunique()} wells")
    
    # Ensure date is datetime
    well_ts = well_ts.copy()
    well_ts[date_col] = pd.to_datetime(well_ts[date_col])
    well_ts = well_ts.sort_values([well_id_col, date_col])
    
    interpolated_list = []
    skipped_wells = 0
    
    for well_id, group in well_ts.groupby(well_id_col):
        # Skip wells with less than 2 observations (minimum for interpolation)
        if len(group) < 2:
            skipped_wells += 1
            continue
        
        # Remove duplicates and sort
        group = group.drop_duplicates(subset=[date_col]).sort_values(date_col)
        
        # Get date range
        start_date = group[date_col].min()
        end_date = group[date_col].max()
        
        # Check gap size if specified
        if max_gap_days is not None:
            date_diffs = group[date_col].diff().dt.days
            max_observed_gap = date_diffs.max()
            if max_observed_gap > max_gap_days:
                logger.debug(f"Well {well_id} has gap > {max_gap_days} days, skipping")
                skipped_wells += 1
                continue
        
        # Generate daily date sequence
        full_dates = pd.date_range(start=start_date, end=end_date, freq=target_frequency)
        
        # Convert dates to ordinal numbers for interpolation
        x_obs = group[date_col].map(pd.Timestamp.toordinal).values
        y_obs = group[wte_col].values
        
        # Remove NaN values
        mask = ~np.isnan(y_obs)
        if mask.sum() < 2:
            skipped_wells += 1
            continue
        
        x_obs = x_obs[mask]
        y_obs = y_obs[mask]
        
        # Perform PCHIP interpolation
        try:
            interpolator = PchipInterpolator(x_obs, y_obs, extrapolate=extrapolate)
            x_new = full_dates.map(pd.Timestamp.toordinal).values
            
            # Only interpolate within data range unless extrapolate=True
            if not extrapolate:
                x_new = x_new[(x_new >= x_obs.min()) & (x_new <= x_obs.max())]
                full_dates = full_dates[(full_dates >= start_date) & (full_dates <= end_date)]
            
            y_new = interpolator(x_new)
            
            # Create interpolated DataFrame
            df_interp = pd.DataFrame({
                well_id_col: well_id,
                date_col: full_dates,
                wte_col: y_new
            })
            
            interpolated_list.append(df_interp)
            
        except Exception as e:
            logger.warning(f"Failed to interpolate well {well_id}: {e}")
            skipped_wells += 1
            continue
    
    if not interpolated_list:
        logger.warning("No wells successfully interpolated")
        return pd.DataFrame(columns=[well_id_col, date_col, wte_col])
    
    result = pd.concat(interpolated_list, ignore_index=True)
    
    logger.info(f"Interpolated {len(interpolated_list)} wells ({skipped_wells} skipped)")
    logger.info(f"Generated {len(result):,} daily interpolated values")
    
    return result

