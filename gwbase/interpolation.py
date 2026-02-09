"""
Time series interpolation functions for GWBASE.

This module implements Step 5 of the GWBASE workflow:
- PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation
  of groundwater levels to **monthly** resolution (values at the middle of each month)
"""

import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm


def interpolate_daily_pchip(
    well_ts: pd.DataFrame,
    well_id_col: str = 'well_id',
    date_col: str = 'date',
    value_col: str = 'wte'
) -> pd.DataFrame:
    """
    Perform **monthly** PCHIP interpolation on groundwater well time series data,
    evaluating at the **middle of each month**.

    PCHIP interpolation preserves monotonic patterns between observations and
    avoids unrealistic oscillations that can occur with traditional spline
    interpolation.

    Parameters
    ----------
    well_ts : pd.DataFrame
        DataFrame containing well measurements with columns:
        - well_id: Well identifier
        - date: Measurement date
        - wte: Water table elevation (or other value to interpolate)
    well_id_col : str, default 'well_id'
        Name of well ID column
    date_col : str, default 'date'
        Name of date column
    value_col : str, default 'wte'
        Name of value column to interpolate

    Returns
    -------
    pd.DataFrame
        DataFrame with **monthly** interpolated values for each well:
        - well_id: Well identifier
        - date: Middle-of-month date
        - wte: Interpolated water table elevation

    Notes
    -----
    - Wells with fewer than 2 observations are skipped
    - Local extrema are preserved
    - The method maintains hydrologic realism

    Example
    -------
    >>> monthly_wte = interpolate_daily_pchip(well_data)
    >>> monthly_wte.to_csv('data/processed/well_pchip_monthly.csv', index=False)
    """
    well_ts = well_ts.copy()
    well_ts[date_col] = pd.to_datetime(well_ts[date_col])
    well_ts = well_ts.sort_values([well_id_col, date_col])

    interpolated_list = []
    wells_processed = 0
    wells_skipped = 0

    print("Performing PCHIP interpolation...")

    for well_id, group in tqdm(well_ts.groupby(well_id_col)):
        # Skip wells with less than 2 observations
        if len(group) < 2:
            wells_skipped += 1
            continue

        # Get date range
        start_date = group[date_col].min()
        end_date = group[date_col].max()

        # Generate monthly date sequence at middle of each month
        # Determine the first and last month covered by the observations
        first_month_start = start_date.to_period('M').to_timestamp()
        last_month_start = end_date.to_period('M').to_timestamp()
        month_starts = pd.date_range(start=first_month_start, end=last_month_start, freq='MS')
        # Middle of month is approximated as 15th (or 14 days after month start)
        full_dates = month_starts + pd.offsets.Day(14)

        # Convert dates to ordinal numbers for interpolation
        x_obs = group[date_col].map(pd.Timestamp.toordinal)
        y_obs = group[value_col].values

        # Perform PCHIP interpolation
        try:
            interpolator = PchipInterpolator(x_obs, y_obs)
            x_new = full_dates.map(pd.Timestamp.toordinal)
            y_new = interpolator(x_new)

            # Create interpolated DataFrame
            df_interp = pd.DataFrame({
                well_id_col: well_id,
                date_col: full_dates,
                value_col: y_new
            })

            interpolated_list.append(df_interp)
            wells_processed += 1

        except Exception as e:
            wells_skipped += 1
            continue

    if interpolated_list:
        interpolated_df = pd.concat(interpolated_list, ignore_index=True)
    else:
        interpolated_df = pd.DataFrame(columns=[well_id_col, date_col, value_col])

    print(f"\nPCHIP Interpolation Summary:")
    print(f"  Wells processed: {wells_processed}")
    print(f"  Wells skipped (< 2 observations): {wells_skipped}")
    print(f"  Total monthly records generated: {len(interpolated_df):,}")

    return interpolated_df


def interpolate_daily(
    well_ts: pd.DataFrame,
    well_id_col: str = 'well_id',
    date_col: str = 'date',
    value_col: str = 'wte'
) -> pd.DataFrame:
    """
    Perform PCHIP interpolation to **daily** resolution.

    Parameters
    ----------
    well_ts : pd.DataFrame
        DataFrame with well_id, date, wte
    well_id_col, date_col, value_col : str
        Column names

    Returns
    -------
    pd.DataFrame
        Daily interpolated values
    """
    well_ts = well_ts.copy()
    well_ts[date_col] = pd.to_datetime(well_ts[date_col])
    well_ts = well_ts.sort_values([well_id_col, date_col])

    interpolated_list = []

    for well_id, group in well_ts.groupby(well_id_col):
        if len(group) < 2:
            continue

        start_date = group[date_col].min().normalize()
        end_date = group[date_col].max().normalize()
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        x_obs = group[date_col].map(pd.Timestamp.toordinal)
        y_obs = group[value_col].values

        try:
            interpolator = PchipInterpolator(x_obs, y_obs)
            x_new = full_dates.map(pd.Timestamp.toordinal)
            y_new = interpolator(x_new)

            interpolated_list.append(pd.DataFrame({
                well_id_col: well_id,
                date_col: full_dates,
                value_col: y_new
            }))
        except Exception:
            continue

    if interpolated_list:
        return pd.concat(interpolated_list, ignore_index=True)
    return pd.DataFrame(columns=[well_id_col, date_col, value_col])


def interpolate_with_well_info(
    well_ts: pd.DataFrame,
    well_info: pd.DataFrame,
    lat_col: str = 'lat_dec',
    lon_col: str = 'long_dec',
    elev_col: str = 'gse'
) -> pd.DataFrame:
    """
    Perform PCHIP interpolation and merge with well location information.

    Parameters
    ----------
    well_ts : pd.DataFrame
        Well time series data
    well_info : pd.DataFrame
        Well metadata with location and elevation
    lat_col : str, default 'lat_dec'
        Latitude column name in well_info
    lon_col : str, default 'long_dec'
        Longitude column name in well_info
    elev_col : str, default 'gse'
        Ground surface elevation column name in well_info

    Returns
    -------
    pd.DataFrame
        Interpolated data merged with well location info

    Example
    -------
    >>> merged = interpolate_with_well_info(well_ts, well_info)
    """
    # Perform interpolation
    daily_interp_df = interpolate_daily_pchip(well_ts)

    # Standardize column names in well_info
    well_info = well_info.copy()
    well_info.columns = well_info.columns.str.lower()

    # Merge with well information
    merged_data = pd.merge(
        daily_interp_df,
        well_info,
        on='well_id',
        how='left'
    )

    # Rename columns for consistency
    rename_mapping = {}
    if lat_col.lower() in merged_data.columns:
        rename_mapping[lat_col.lower()] = 'well_lat'
    if lon_col.lower() in merged_data.columns:
        rename_mapping[lon_col.lower()] = 'well_lon'

    if rename_mapping:
        merged_data = merged_data.rename(columns=rename_mapping)

    # Select key columns
    keep_cols = ['well_id', 'date', 'wte', 'well_lat', 'well_lon']
    if elev_col.lower() in merged_data.columns:
        keep_cols.append(elev_col.lower())

    available_cols = [c for c in keep_cols if c in merged_data.columns]
    merged_data = merged_data[available_cols]

    print(f"Merged data shape: {merged_data.shape}")

    return merged_data


def validate_interpolation(
    original: pd.DataFrame,
    interpolated: pd.DataFrame,
    well_id_col: str = 'well_id',
    value_col: str = 'wte'
) -> dict:
    """
    Validate interpolation results against original observations.

    Parameters
    ----------
    original : pd.DataFrame
        Original well observations
    interpolated : pd.DataFrame
        Interpolated daily values
    well_id_col : str
        Well ID column name
    value_col : str
        Value column name

    Returns
    -------
    dict
        Validation statistics

    Example
    -------
    >>> stats = validate_interpolation(original_data, interpolated_data)
    """
    # Calculate statistics
    original_stats = original.groupby(well_id_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max'])
    interp_stats = interpolated.groupby(well_id_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max'])

    # Merge for comparison
    comparison = original_stats.join(
        interp_stats,
        lsuffix='_orig',
        rsuffix='_interp'
    )

    validation = {
        'n_wells_original': len(original_stats),
        'n_wells_interpolated': len(interp_stats),
        'total_original_records': original_stats['count'].sum(),
        'total_interpolated_records': interp_stats['count'].sum(),
        'interpolation_ratio': interp_stats['count'].sum() / original_stats['count'].sum(),
        'mean_error': (comparison['mean_interp'] - comparison['mean_orig']).abs().mean(),
    }

    print("\nInterpolation Validation:")
    for key, value in validation.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return validation
