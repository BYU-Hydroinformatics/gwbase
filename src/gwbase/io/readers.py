"""Data reading utilities."""
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_groundwater_data(csv_path: Path) -> pd.DataFrame:
    """
    Load groundwater time series data.
    
    Parameters:
    -----------
    csv_path : Path
        Path to groundwater CSV file
        
    Returns:
    --------
    DataFrame
        Groundwater data with standardized column names
    """
    logger.info(f"Loading groundwater data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Standardize column names - build mapping dict first
    column_mapping = {}
    
    # Map uppercase/lowercase variations
    if 'Well_ID' in df.columns:
        column_mapping['Well_ID'] = 'well_id'
    elif 'well_id' not in df.columns:
        for col in df.columns:
            if col.lower() in ['well_id', 'wellid']:
                column_mapping[col] = 'well_id'
                break
    
    if 'Date' in df.columns:
        column_mapping['Date'] = 'date'
    elif 'date' not in df.columns:
        for col in df.columns:
            if col.lower() == 'date':
                column_mapping[col] = 'date'
                break
    
    if 'WTE' in df.columns:
        column_mapping['WTE'] = 'wte'
    elif 'wte' not in df.columns:
        for col in df.columns:
            if col.lower() == 'wte':
                column_mapping[col] = 'wte'
                break
    
    # Apply renaming only if we have mappings
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded {len(df)} groundwater records")
    return df


def load_hydrography_data(
    gages_csv: Path,
    streams_shp: Path,
    subbasin_shp: Path,
    wells_shp: Path
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load all hydrography spatial data.
    
    Returns:
    --------
    Tuple of (gages_gdf, streams_gdf, subbasin_gdf, wells_gdf)
    """
    logger.info("Loading hydrography data")
    
    # Load gages
    gage_df = pd.read_csv(gages_csv)
    subbasin_gdf = gpd.read_file(subbasin_shp)
    
    # Convert gages to GeoDataFrame
    gages_gdf = gpd.GeoDataFrame(
        gage_df,
        geometry=gpd.points_from_xy(gage_df['longitude'], gage_df['latitude']),
        crs=subbasin_gdf.crs
    )
    
    # Load streams and wells
    streams_gdf = gpd.read_file(streams_shp)
    wells_gdf = gpd.read_file(wells_shp)
    
    logger.info(f"Loaded {len(gages_gdf)} gages, {len(streams_gdf)} stream reaches, "
                f"{len(subbasin_gdf)} catchments, {len(wells_gdf)} wells")
    
    return gages_gdf, streams_gdf, subbasin_gdf, wells_gdf


def load_streamflow_data(
    streamflow_dir: Path,
    bfd_column: str = "ML_BFD",
    bfd_value: int = 1
) -> pd.DataFrame:
    """
    Load and compile streamflow data from directory of CSV files.
    
    Parameters:
    -----------
    streamflow_dir : Path
        Directory containing gage CSV files
    bfd_column : str
        Column name for BFD flag
    bfd_value : int
        Value indicating baseflow-dominated day
        
    Returns:
    --------
    DataFrame
        Compiled streamflow data with columns: gage_id, date, q, bfd
    """
    logger.info(f"Loading streamflow data from {streamflow_dir}")
    
    compiled = []
    for csv_file in streamflow_dir.glob("*.csv"):
        gage_id = csv_file.stem
        
        df = pd.read_csv(csv_file)
        df = df.copy()  # Avoid SettingWithCopyWarning
        
        # Filter for BFD=1 if column exists
        if bfd_column in df.columns:
            df = df[df[bfd_column] == bfd_value].copy()
            df['bfd'] = df[bfd_column]
        else:
            df['bfd'] = 1  # Assume all are BFD if no column
        
        # Select and rename columns
        column_mapping = {'Q': 'q'}
        if 'date' not in df.columns:
            # Try common date column names
            for col in df.columns:
                if col.lower() in ['date', 'datetime', 'time']:
                    column_mapping[col] = 'date'
                    break
        
        df = df.rename(columns=column_mapping)
        df['gage_id'] = gage_id
        df = df[['gage_id', 'date', 'q', 'bfd']].copy()
        
        compiled.append(df)
    
    if not compiled:
        logger.warning(f"No streamflow CSV files found in {streamflow_dir}")
        return pd.DataFrame(columns=['gage_id', 'date', 'q', 'bfd'])
    
    result = pd.concat(compiled, ignore_index=True)
    
    if 'date' in result.columns:
        result['date'] = pd.to_datetime(result['date'])
    
    logger.info(f"Loaded streamflow data for {result['gage_id'].nunique()} gages")
    return result
