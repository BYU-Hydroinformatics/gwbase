"""Well-reach association and elevation functions."""
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def find_nearest_reach_and_elevation(
    wells_gdf: gpd.GeoDataFrame,
    streams_gdf: gpd.GeoDataFrame,
    reach_elevation_df: Optional[pd.DataFrame] = None,
    elevation_column: str = "Avg_GSE",
    reach_id_column: str = "Reach_ID"
) -> pd.DataFrame:
    """
    Find nearest stream reach for each well and record reach ID and elevation.
    
    Parameters:
    -----------
    wells_gdf : GeoDataFrame
        Wells with geometry column
    streams_gdf : GeoDataFrame
        Stream network with LINKNO column
    reach_elevation_df : DataFrame, optional
        Reach elevation data with Reach_ID and elevation column
    elevation_column : str
        Column name for elevation in reach_elevation_df
    reach_id_column : str
        Column name for reach ID in reach_elevation_df
        
    Returns:
    --------
    DataFrame
        Columns: well_id, reach_id, reach_elevation_m, distance_to_reach_m
    """
    logger.info("Finding nearest reach for each well")
    
    # Ensure CRS match - convert to UTM for distance calculations
    target_crs = 'EPSG:32612'  # UTM Zone 12N (common for western US)
    
    if wells_gdf.crs is None:
        logger.warning("Wells have no CRS, assuming EPSG:4326")
        wells_gdf = wells_gdf.set_crs('EPSG:4326')
    
    if streams_gdf.crs is None:
        logger.warning("Streams have no CRS, assuming EPSG:4326")
        streams_gdf = streams_gdf.set_crs('EPSG:4326')
    
    # Convert to UTM for accurate distance calculations
    wells_utm = wells_gdf.to_crs(target_crs)
    streams_utm = streams_gdf.to_crs(target_crs)
    
    # Find linkno column
    linkno_col = None
    for col in ['LINKNO', 'linkno', 'LINK_NO']:
        if col in streams_utm.columns:
            linkno_col = col
            break
    
    if linkno_col is None:
        raise ValueError("Stream GeoDataFrame must have LINKNO column")
    
    # Find well_id column
    well_id_col = None
    for col in ['well_id', 'WELL_ID', 'Well_ID', 'wellid', 'WellID']:
        if col in wells_utm.columns:
            well_id_col = col
            break
    
    if well_id_col is None:
        # Try to find any column with 'well' in name
        for col in wells_utm.columns:
            if 'well' in col.lower():
                well_id_col = col
                break
    
    if well_id_col is None:
        raise ValueError("Cannot find well_id column in wells GeoDataFrame")
    
    results = []
    
    # For each well, find nearest stream reach
    for idx, well in wells_utm.iterrows():
        well_id = well[well_id_col]
        well_geom = well.geometry
        
        if well_geom is None or well_geom.is_empty:
            continue
        
        # Calculate distances to all stream reaches
        distances = streams_utm.geometry.distance(well_geom)
        nearest_idx = distances.idxmin()
        nearest_reach = streams_utm.iloc[nearest_idx]
        
        reach_id = int(nearest_reach[linkno_col])
        distance_m = float(distances.iloc[nearest_idx])
        
        # Get reach elevation if available
        reach_elevation = None
        if reach_elevation_df is not None:
            reach_elev_data = reach_elevation_df[
                reach_elevation_df[reach_id_column] == reach_id
            ]
            if not reach_elev_data.empty:
                elev_val = reach_elev_data.iloc[0][elevation_column]
                # Convert feet to meters if needed (assuming elevation in feet)
                if pd.notna(elev_val):
                    reach_elevation = float(elev_val) * 0.3048  # Convert to meters
        
        results.append({
            'well_id': well_id,
            'reach_id': reach_id,
            'reach_elevation_m': reach_elevation,
            'distance_to_reach_m': distance_m
        })
    
    result_df = pd.DataFrame(results)
    logger.info(f"Found nearest reaches for {len(result_df)} wells")
    logger.info(f"  {result_df['reach_elevation_m'].notna().sum()} wells have reach elevation data")
    
    return result_df

