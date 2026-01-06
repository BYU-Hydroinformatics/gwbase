"""Pairing groundwater and streamflow data."""
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def pair_wte_with_streamflow_bfd(
    wte_data: pd.DataFrame,
    streamflow_data: pd.DataFrame,
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    date_col: str = 'date',
    wte_col: str = 'wte',
    q_col: str = 'q',
    bfd_col: str = 'bfd',
    bfd_value: int = 1
) -> pd.DataFrame:
    """
    Pair daily groundwater (WTE) and streamflow data on baseflow-dominated days.
    
    Parameters:
    -----------
    wte_data : DataFrame
        Daily interpolated WTE data with columns: well_id, gage_id, date, wte
    streamflow_data : DataFrame
        Streamflow data with columns: gage_id, date, q, bfd
    well_id_col : str
        Column name for well ID
    gage_id_col : str
        Column name for gage ID
    date_col : str
        Column name for date
    wte_col : str
        Column name for water table elevation
    q_col : str
        Column name for streamflow
    bfd_col : str
        Column name for BFD flag
    bfd_value : int
        Value indicating baseflow-dominated day
        
    Returns:
    --------
    DataFrame
        Paired data with columns: well_id, gage_id, date, wte, q, bfd
    """
    logger.info("Pairing WTE with streamflow on BFD days")
    
    wte_data = wte_data.copy()
    streamflow_data = streamflow_data.copy()
    
    wte_data[date_col] = pd.to_datetime(wte_data[date_col])
    streamflow_data[date_col] = pd.to_datetime(streamflow_data[date_col])
    
    # Filter streamflow to BFD days only
    streamflow_bfd = streamflow_data[
        streamflow_data[bfd_col] == bfd_value
    ].copy()
    
    # Merge on gage_id and date
    paired = pd.merge(
        wte_data,
        streamflow_bfd[[gage_id_col, date_col, q_col, bfd_col]],
        on=[gage_id_col, date_col],
        how='inner'
    )
    
    logger.info(f"Paired data: {len(paired):,} records")
    return paired

