"""Elevation-based filtering."""
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def filter_by_elevation_buffer(
    well_data: pd.DataFrame,
    reach_data: pd.DataFrame,
    vertical_buffer_meters: float = 30.0,
    wte_units_feet: bool = True,
    conversion_factor: float = 0.3048,
    wte_col: str = 'wte',
    reach_elev_col: str = 'reach_elevation_m',
    well_id_col: str = 'well_id'
) -> pd.DataFrame:
    """
    Filter wells based on elevation buffer relative to nearest reach.
    
    Parameters:
    -----------
    well_data : DataFrame
        Well data with wte column
    reach_data : DataFrame
        Reach data with reach_elevation_m column and well_id
    vertical_buffer_meters : float
        Maximum vertical distance below reach elevation allowed
    wte_units_feet : bool
        If True, convert WTE from feet to meters
    conversion_factor : float
        Conversion factor (feet to meters = 0.3048)
    wte_col : str
        Column name for water table elevation
    reach_elev_col : str
        Column name for reach elevation in meters
    well_id_col : str
        Column name for well ID
        
    Returns:
    --------
    DataFrame
        Filtered well data
    """
    logger.info(f"Filtering wells by elevation buffer: {vertical_buffer_meters} m")
    
    # Merge well and reach data
    merged = pd.merge(
        well_data,
        reach_data[[well_id_col, reach_elev_col]],
        on=well_id_col,
        how='inner'
    )
    
    # Convert WTE to meters if needed
    if wte_units_feet:
        merged['wte_meters'] = merged[wte_col] * conversion_factor
    else:
        merged['wte_meters'] = merged[wte_col]
    
    # Calculate vertical distance (reach elevation - WTE)
    merged['delta_elev'] = merged[reach_elev_col] - merged['wte_meters']
    
    # Filter: keep if WTE >= reach_elev OR within buffer
    filtered = merged[merged['delta_elev'] <= vertical_buffer_meters].copy()
    
    # Remove temporary columns
    if 'wte_meters' in filtered.columns:
        filtered = filtered.drop(columns=['wte_meters'])
    
    logger.info(f"Elevation filtering: {len(merged):,} -> {len(filtered):,} records")
    return filtered

