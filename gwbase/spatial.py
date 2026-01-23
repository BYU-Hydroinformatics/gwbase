"""
Spatial analysis functions for GWBASE.

This module implements Steps 2-3 of the GWBASE workflow:
- Locate groundwater wells within catchments (Step 2)
- Associate wells with nearest stream segments (Step 3)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from typing import Optional
from .network import find_downstream_gage


def extract_reach_centroids(stream_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extract centroids from stream reaches.

    Parameters
    ----------
    stream_gdf : gpd.GeoDataFrame
        Stream network GeoDataFrame with LINKNO column

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Reach_ID, Latitude, Longitude

    Example
    -------
    >>> centroids = extract_reach_centroids(stream_gdf)
    >>> centroids.to_csv('data/processed/reach_centroids.csv', index=False)
    """
    # Ensure the geometry is in geographic coordinates (WGS84)
    if stream_gdf.crs != "EPSG:4326":
        stream_gdf = stream_gdf.to_crs("EPSG:4326")

    centroids = []
    print("Extracting reach centroids...")

    for reach in tqdm(stream_gdf.itertuples(), total=len(stream_gdf)):
        centroid = reach.geometry.centroid
        centroids.append({
            'Reach_ID': reach.LINKNO,
            'Latitude': centroid.y,
            'Longitude': centroid.x
        })

    centroids_df = pd.DataFrame(centroids)
    print(f"Extracted {len(centroids_df)} reach centroids")

    return centroids_df


def locate_wells_in_catchments(
    wells_gdf: gpd.GeoDataFrame,
    catchment_gdf: gpd.GeoDataFrame,
    upstream_mapping: pd.DataFrame
) -> pd.DataFrame:
    """
    Locate groundwater wells within catchments and assign to terminal gages.

    Parameters
    ----------
    wells_gdf : gpd.GeoDataFrame
        GeoDataFrame of well locations
    catchment_gdf : gpd.GeoDataFrame
        GeoDataFrame of catchment polygons with linkno column
    upstream_mapping : pd.DataFrame
        DataFrame from delineate_all_upstream_catchments with
        Gage_ID, Gage_Name, Upstream_Catchment_ID columns

    Returns
    -------
    pd.DataFrame
        DataFrame with well-gage associations

    Example
    -------
    >>> wells_with_gages = locate_wells_in_catchments(wells_gdf, catchment_gdf, upstream_df)
    """
    # Find linkno column
    linkno_col = 'linkno' if 'linkno' in catchment_gdf.columns else 'LINKNO'

    # Ensure consistent CRS
    if wells_gdf.crs != catchment_gdf.crs:
        wells_gdf = wells_gdf.to_crs(catchment_gdf.crs)

    # Spatial join to find which catchment each well belongs to
    wells_in_catchments = gpd.sjoin(
        wells_gdf,
        catchment_gdf[[linkno_col, 'geometry']],
        how='inner',
        predicate='within'
    ).rename(columns={linkno_col: 'catchment_id'})

    # Merge with upstream mapping to get gage assignments
    wells_in_catchments = wells_in_catchments.merge(
        upstream_mapping[['Gage_ID', 'Gage_Name', 'Upstream_Catchment_ID']],
        left_on='catchment_id',
        right_on='Upstream_Catchment_ID',
        how='left'
    )

    # Standardize column names
    wells_in_catchments.columns = wells_in_catchments.columns.str.lower()

    # Drop rows without gage assignment
    wells_with_gages = wells_in_catchments.dropna(subset=['gage_id'])

    print(f"Located {wells_with_gages['well_id'].nunique()} wells in {wells_with_gages['gage_id'].nunique()} gage watersheds")

    return wells_with_gages


def associate_wells_with_reaches(
    wells_gdf: gpd.GeoDataFrame,
    stream_gdf: gpd.GeoDataFrame,
    gages_df: pd.DataFrame,
    reach_elevations: pd.DataFrame,
    utm_crs: str = "EPSG:32612"
) -> pd.DataFrame:
    """
    Associate each well with its nearest stream reach and compute distances.

    Parameters
    ----------
    wells_gdf : gpd.GeoDataFrame
        GeoDataFrame of well locations
    stream_gdf : gpd.GeoDataFrame
        GeoDataFrame of stream network
    gages_df : pd.DataFrame
        DataFrame with gage information including COMID_v2 for reach matching
    reach_elevations : pd.DataFrame
        DataFrame with reach elevations (Reach_ID, Avg_GSE columns)
    utm_crs : str, default "EPSG:32612"
        UTM projection for accurate distance calculations

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Well_ID: Well identifier
        - Reach_ID: Nearest reach identifier
        - Reach_Elevation: Elevation of nearest reach (m)
        - Distance_to_Reach: Distance to nearest reach (m)
        - Downstream_Gage: ID of downstream gage (if any)

    Example
    -------
    >>> well_reach = associate_wells_with_reaches(wells_gdf, stream_gdf, gages_df, reach_elev)
    """
    # Convert to UTM for accurate distance calculations
    print("Converting coordinate systems to UTM...")
    wells_utm = wells_gdf.to_crs(utm_crs)
    stream_utm = stream_gdf.to_crs(utm_crs)

    print(f"Processing {len(wells_utm)} wells...")

    results = []

    for well in tqdm(wells_utm.itertuples(), total=len(wells_utm)):
        # Calculate distances to all stream segments
        distances = stream_utm.geometry.distance(well.geometry)
        nearest_reach_idx = distances.idxmin()
        nearest_reach = stream_utm.iloc[nearest_reach_idx]
        min_distance = distances.min()

        # Get reach elevation
        reach_id = nearest_reach['LINKNO']
        reach_elev_data = reach_elevations[reach_elevations['Reach_ID'] == reach_id]
        reach_elevation = reach_elev_data.iloc[0]['Avg_GSE'] if not reach_elev_data.empty else None

        # Find downstream gage
        downstream_gage = find_downstream_gage(
            reach_id,
            stream_gdf,
            gages_df
        )

        results.append({
            'Well_ID': well.Well_ID,
            'Reach_ID': reach_id,
            'Reach_Elevation': reach_elevation,
            'Distance_to_Reach': min_distance,
            'Downstream_Gage': downstream_gage
        })

    results_df = pd.DataFrame(results)

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total wells processed: {len(results_df)}")
    print(f"Wells with downstream gages: {results_df['Downstream_Gage'].notna().sum()}")
    print(f"Wells with reach elevations: {results_df['Reach_Elevation'].notna().sum()}")
    print(f"\nDistance statistics (meters):")
    print(f"  Minimum: {results_df['Distance_to_Reach'].min():.2f} m")
    print(f"  Maximum: {results_df['Distance_to_Reach'].max():.2f} m")
    print(f"  Average: {results_df['Distance_to_Reach'].mean():.2f} m")
    print(f"  Median: {results_df['Distance_to_Reach'].median():.2f} m")

    return results_df


def merge_well_reach_data(
    well_pchip: pd.DataFrame,
    well_reach: pd.DataFrame,
    gage_to_wells: pd.DataFrame,
    gage_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge well time series with reach and gage information.

    Parameters
    ----------
    well_pchip : pd.DataFrame
        Interpolated well time series with well_id, date, wte columns
    well_reach : pd.DataFrame
        Well-reach associations from associate_wells_with_reaches
    gage_to_wells : pd.DataFrame
        Well-gage assignments from locate_wells_in_catchments
    gage_info : pd.DataFrame
        Gage metadata with id, latitude, longitude columns

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all well, reach, and gage information

    Example
    -------
    >>> merged = merge_well_reach_data(well_pchip, well_reach, gage_wells, gage_info)
    """
    # Merge well time series with gage assignments
    merged = pd.merge(
        well_pchip,
        gage_to_wells[['gage_id', 'well_id', 'well_lat', 'well_lon']],
        on='well_id',
        how='inner'
    )

    # Add gage coordinates
    merged = pd.merge(
        merged,
        gage_info[['id', 'latitude', 'longitude']],
        left_on='gage_id',
        right_on='id',
        how='left'
    ).rename(columns={
        'latitude': 'gage_lat',
        'longitude': 'gage_lon'
    }).drop('id', axis=1)

    # Add reach elevation for filtering
    merged = pd.merge(
        merged,
        well_reach[['Well_ID', 'Reach_Elevation']].rename(columns={
            'Well_ID': 'well_id',
            'Reach_Elevation': 'reach_elev_m'
        }),
        on='well_id',
        how='left'
    )

    print(f"Merged dataset: {len(merged):,} records")
    print(f"  Unique wells: {merged['well_id'].nunique()}")
    print(f"  Unique gages: {merged['gage_id'].nunique()}")

    return merged
