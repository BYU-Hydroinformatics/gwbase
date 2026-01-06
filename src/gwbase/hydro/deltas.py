"""Delta metrics calculation."""
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def calculate_delta_metrics(
    paired_data: pd.DataFrame,
    baseline_method: str = "first_bfd",
    well_id_col: str = 'well_id',
    gage_id_col: str = 'gage_id',
    date_col: str = 'date',
    wte_col: str = 'wte',
    q_col: str = 'q',
    bfd_col: str = 'bfd',
    delta_wte_col: str = 'delta_wte',
    delta_q_col: str = 'delta_q'
) -> pd.DataFrame:
    """
    Calculate delta metrics (change from baseline).
    
    Parameters:
    -----------
    paired_data : DataFrame
        Paired WTE-streamflow data
    baseline_method : str
        Method for determining baseline
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
    delta_wte_col : str
        Output column name for delta WTE
    delta_q_col : str
        Output column name for delta Q
        
    Returns:
    --------
    DataFrame
        Data with delta metrics added
    """
    logger.info(f"Calculating delta metrics using baseline method: {baseline_method}")
    
    result = paired_data.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Calculate baseline for each well-gage pair
    baselines = []
    
    for (well_id, gage_id), group in result.groupby([well_id_col, gage_id_col]):
        group_sorted = group.sort_values(date_col)
        
        if baseline_method == "first_bfd":
            first_bfd = group_sorted.iloc[0]
            wte_baseline = first_bfd[wte_col]
            q_baseline = first_bfd[q_col]
        elif baseline_method == "mean":
            wte_baseline = group[wte_col].mean()
            q_baseline = group[q_col].mean()
        elif baseline_method == "median":
            wte_baseline = group[wte_col].median()
            q_baseline = group[q_col].median()
        else:
            raise ValueError(f"Unknown baseline method: {baseline_method}")
        
        baselines.append({
            well_id_col: well_id,
            gage_id_col: gage_id,
            'wte_baseline': wte_baseline,
            'q_baseline': q_baseline
        })
    
    baselines_df = pd.DataFrame(baselines)
    
    # Merge baselines and calculate deltas
    result = pd.merge(
        result,
        baselines_df,
        on=[well_id_col, gage_id_col],
        how='left'
    )
    
    result[delta_wte_col] = result[wte_col] - result['wte_baseline']
    result[delta_q_col] = result[q_col] - result['q_baseline']
    
    result = result.drop(columns=['wte_baseline', 'q_baseline'])
    
    logger.info(f"Calculated delta metrics for {len(result):,} records")
    return result

