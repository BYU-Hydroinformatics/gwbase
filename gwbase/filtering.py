"""
Elevation-based filtering functions for GWBASE.

This module implements Step 6 of the GWBASE workflow:
- Filter wells based on vertical separation from stream elevation
- Retain wells with water levels close to or above streambed elevation
"""

import pandas as pd
import numpy as np
from typing import Tuple


def filter_by_elevation(
    filtered_data: pd.DataFrame,
    well_reach_df: pd.DataFrame,
    distance_buffer_meters: float = 30.0,
    wte_feet_to_meters: float = 0.3048
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter wells based on elevation difference between WTE and stream reach.

    The logic is that wells with water levels far below the stream elevation
    are not likely to impact baseflow to the stream. This filter retains wells
    where groundwater levels are close to or higher than the nearby stream
    channel, reflecting conditions that can support hydrologic exchange.

    Parameters
    ----------
    filtered_data : pd.DataFrame
        DataFrame with well time series including 'wte' column (in feet)
    well_reach_df : pd.DataFrame
        DataFrame with well-reach associations including 'reach_elev_m' column
    distance_buffer_meters : float, default 30.0
        Maximum allowed vertical distance below streambed (meters)
    wte_feet_to_meters : float, default 0.3048
        Conversion factor from feet to meters

    Returns
    -------
    tuple
        (filtered_result, dist_stats) - Filtered data and statistics

    Notes
    -----
    - Wells with WTE > reach elevation (gaining stream) are always kept
    - Wells with WTE within buffer distance below reach elevation are kept
    - Wells with WTE more than buffer distance below reach elevation are excluded

    Example
    -------
    >>> filtered, stats = filter_by_elevation(data, well_reach, distance_buffer_meters=30)
    >>> print(f"Retained {len(filtered):,} records")
    """
    filtered_data = filtered_data.copy()

    # Convert WTE from feet to meters
    filtered_data['wte_meters'] = filtered_data['wte'] * wte_feet_to_meters

    # Merge with well-reach data to get reach elevation
    merged_data = pd.merge(
        filtered_data,
        well_reach_df[['well_id', 'reach_elev_m']],
        on='well_id',
        how='inner'
    )

    # Calculate elevation difference (reach - WTE)
    # Positive values = WTE below streambed (losing/disconnected)
    # Negative values = WTE above streambed (gaining stream)
    merged_data['delta_elev'] = merged_data['reach_elev_m'] - merged_data['wte_meters']

    # Filter: keep wells where delta_elev <= buffer distance
    # This keeps:
    #   - All gaining stream conditions (negative delta_elev)
    #   - Wells within buffer distance below streambed
    filtered_result = merged_data[
        merged_data['delta_elev'] <= distance_buffer_meters
    ].copy()

    # Define delta bins for statistics
    delta_bins = [
        -float('inf'), -20, -10, -5, 0, 5, 10, 20, 30, 50, 75, 100, float('inf')
    ]
    bin_labels = [
        "< -20", "-20 to -10", "-10 to -5", "-5 to 0",
        "0 to 5", "5 to 10", "10 to 20", "20 to 30",
        "30 to 50", "50 to 75", "75 to 100", ">= 100"
    ]

    # Bin the delta_elev values
    filtered_result['delta_bin'] = pd.cut(
        filtered_result['delta_elev'],
        bins=delta_bins,
        labels=bin_labels
    )

    # Calculate statistics
    total_measurements = len(filtered_result)
    dist_stats = filtered_result.groupby('delta_bin', observed=True).size().reset_index(name='count')
    dist_stats['percentage'] = (dist_stats['count'] / total_measurements * 100).round(2)

    # Print summary
    print(f"\nElevation-Based Filtering (buffer = {distance_buffer_meters}m):")
    print(f"  Input records: {len(merged_data):,}")
    print(f"  Filtered records: {len(filtered_result):,}")
    print(f"  Retention rate: {len(filtered_result)/len(merged_data)*100:.1f}%")
    print(f"\n  Delta elevation distribution:")
    print(dist_stats.to_string(index=False))

    return filtered_result, dist_stats


def analyze_elevation_sensitivity(
    filtered_data: pd.DataFrame,
    well_reach_df: pd.DataFrame,
    buffer_values: list = None
) -> pd.DataFrame:
    """
    Analyze sensitivity of results to different elevation buffer values.

    Parameters
    ----------
    filtered_data : pd.DataFrame
        Input data with well time series
    well_reach_df : pd.DataFrame
        Well-reach elevation data
    buffer_values : list, optional
        List of buffer values to test. Default: [10, 20, 30, 50, 100]

    Returns
    -------
    pd.DataFrame
        Summary statistics for each buffer value

    Example
    -------
    >>> sensitivity = analyze_elevation_sensitivity(data, well_reach)
    """
    if buffer_values is None:
        buffer_values = [10, 20, 30, 50, 100]

    results = []

    for buffer in buffer_values:
        filtered, _ = filter_by_elevation(
            filtered_data,
            well_reach_df,
            distance_buffer_meters=buffer
        )

        results.append({
            'buffer_m': buffer,
            'n_records': len(filtered),
            'n_wells': filtered['well_id'].nunique(),
            'n_gages': filtered['gage_id'].nunique() if 'gage_id' in filtered.columns else None,
            'retention_pct': len(filtered) / len(filtered_data) * 100
        })

    sensitivity_df = pd.DataFrame(results)

    print("\nElevation Buffer Sensitivity Analysis:")
    print(sensitivity_df.to_string(index=False))

    return sensitivity_df


def calculate_hydraulic_gradient(
    data: pd.DataFrame,
    wte_col: str = 'wte_meters',
    reach_elev_col: str = 'reach_elev_m',
    distance_col: str = 'distance_to_reach'
) -> pd.DataFrame:
    """
    Calculate hydraulic gradient between wells and nearest streams.

    Parameters
    ----------
    data : pd.DataFrame
        Data with WTE, reach elevation, and distance
    wte_col : str
        Column name for water table elevation (meters)
    reach_elev_col : str
        Column name for reach elevation (meters)
    distance_col : str
        Column name for horizontal distance (meters)

    Returns
    -------
    pd.DataFrame
        Data with hydraulic gradient columns added

    Example
    -------
    >>> data_with_gradient = calculate_hydraulic_gradient(filtered_data)
    """
    result = data.copy()

    # Calculate head difference (positive = towards stream)
    result['head_diff_m'] = result[wte_col] - result[reach_elev_col]

    # Calculate hydraulic gradient (dimensionless)
    # Positive gradient = flow towards stream
    if distance_col in result.columns:
        result['hydraulic_gradient'] = np.where(
            result[distance_col] > 0,
            result['head_diff_m'] / result[distance_col],
            np.nan
        )

    # Classify flow direction
    result['flow_direction'] = np.where(
        result['head_diff_m'] > 0,
        'gaining',
        np.where(result['head_diff_m'] < 0, 'losing', 'neutral')
    )

    # Summary statistics
    flow_summary = result['flow_direction'].value_counts()
    print("\nFlow Direction Summary:")
    print(flow_summary)

    if 'hydraulic_gradient' in result.columns:
        print(f"\nHydraulic Gradient Statistics:")
        print(f"  Mean: {result['hydraulic_gradient'].mean():.6f}")
        print(f"  Median: {result['hydraulic_gradient'].median():.6f}")
        print(f"  Range: {result['hydraulic_gradient'].min():.6f} to {result['hydraulic_gradient'].max():.6f}")

    return result
