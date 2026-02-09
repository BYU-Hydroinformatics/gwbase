"""
Data loading functions for GWBASE.

This module handles loading of:
- Hydrography data (streams, catchments, lakes, basin boundaries)
- Groundwater well data and measurements (CSV format from USGS NWIS)
- Streamflow data from USGS gages
- Gage location and metadata
- Baseflow classification labels
"""

import os
import pandas as pd
import geopandas as gpd
import warnings
from typing import Tuple, Optional

warnings.filterwarnings('ignore')


def load_hydrography_data(filepath: str, layer: str = None) -> gpd.GeoDataFrame:
    """
    Load a hydrography shapefile or GeoPackage (streams, catchments, etc.).

    Parameters
    ----------
    filepath : str
        Path to shapefile (.shp) or GeoPackage (.gpkg)
    layer : str, optional
        Layer name for multi-layer files (GeoPackage)

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with geometry and attributes

    Example
    -------
    >>> streams = load_hydrography_data('data/raw/hydrography/streams.shp')
    >>> catchments = load_hydrography_data('data/catchments.gpkg', layer='catchments')
    """
    if layer:
        gdf = gpd.read_file(filepath, layer=layer)
    else:
        gdf = gpd.read_file(filepath)

    print(f"Loaded {filepath}: {len(gdf)} features")

    return gdf


def load_well_locations(
    filepath: str,
    well_id_col: str = 'Well_ID',
    lat_col: str = 'lat_dec',
    lon_col: str = 'long_dec',
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Load groundwater well locations from a CSV file (e.g., from USGS NWIS).

    Creates a GeoDataFrame with point geometries from lat/lon coordinates.

    Parameters
    ----------
    filepath : str
        Path to CSV file containing well locations.
        Expected columns: Well_ID, lat_dec, long_dec, GSE, and optionally
        Well_Name, AquiferID, Aquifer_Name, State
    well_id_col : str, default 'Well_ID'
        Column name for well identifier
    lat_col : str, default 'lat_dec'
        Column name for latitude (decimal degrees)
    lon_col : str, default 'long_dec'
        Column name for longitude (decimal degrees)
    crs : str, default "EPSG:4326"
        Coordinate reference system (WGS84 geographic)

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with well locations as point geometries

    Example
    -------
    >>> wells_gdf = load_well_locations('data/raw/groundwater/wells.csv')
    >>> print(f"Loaded {len(wells_gdf)} wells")
    """
    # Load CSV
    df = pd.read_csv(filepath)

    # Standardize column names (handle case variations)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == well_id_col.lower():
            col_mapping[col] = 'well_id'
        elif col_lower == lat_col.lower():
            col_mapping[col] = 'lat_dec'
        elif col_lower == lon_col.lower():
            col_mapping[col] = 'long_dec'
        elif col_lower == 'gse':
            col_mapping[col] = 'gse'

    df = df.rename(columns=col_mapping)

    # Validate required columns
    required = ['well_id', 'lat_dec', 'long_dec']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove rows with missing coordinates
    df = df.dropna(subset=['lat_dec', 'long_dec'])

    # Create GeoDataFrame
    wells_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['long_dec'], df['lat_dec']),
        crs=crs
    )

    print(f"Loaded well locations: {len(wells_gdf)} wells")
    print(f"  Coordinate range: ({wells_gdf['lat_dec'].min():.2f}, {wells_gdf['long_dec'].min():.2f}) to "
          f"({wells_gdf['lat_dec'].max():.2f}, {wells_gdf['long_dec'].max():.2f})")

    return wells_gdf


def load_groundwater_data(
    well_locations_path: str,
    timeseries_path: str,
    well_id_col: str = 'Well_ID',
    lat_col: str = 'lat_dec',
    lon_col: str = 'long_dec'
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load groundwater well locations and time series data from CSV files.

    This function loads well data downloaded from USGS NWIS or similar sources.
    It expects two CSV files: one with well locations/metadata and one with
    water level time series.

    Parameters
    ----------
    well_locations_path : str
        Path to CSV file containing well locations and metadata.
        Expected columns: Well_ID, lat_dec, long_dec, GSE, Well_Name, State
    timeseries_path : str
        Path to CSV file containing water level time series.
        Expected columns: well_id (or Well_ID), date, wte
    well_id_col : str, default 'Well_ID'
        Column name for well identifier in locations file
    lat_col : str, default 'lat_dec'
        Column name for latitude
    lon_col : str, default 'long_dec'
        Column name for longitude

    Returns
    -------
    tuple
        (wells_gdf, well_ts, well_info):
        - wells_gdf: GeoDataFrame with well point geometries
        - well_ts: DataFrame with water level time series
        - well_info: DataFrame with well metadata (same data as wells_gdf without geometry)

    Example
    -------
    >>> wells_gdf, well_ts, well_info = load_groundwater_data(
    ...     'data/raw/groundwater/wells.csv',
    ...     'data/raw/groundwater/water_levels.csv'
    ... )
    """
    # Load well locations as GeoDataFrame
    wells_gdf = load_well_locations(
        well_locations_path,
        well_id_col=well_id_col,
        lat_col=lat_col,
        lon_col=lon_col
    )

    # Create well_info DataFrame (without geometry)
    well_info = wells_gdf.drop(columns='geometry').copy()

    # Load time series data
    well_ts = pd.read_csv(timeseries_path)

    # Standardize column names
    col_mapping = {}
    for col in well_ts.columns:
        col_lower = col.lower()
        if col_lower in ['well_id', 'wellid', 'site_no']:
            col_mapping[col] = 'well_id'
        elif col_lower == 'date':
            col_mapping[col] = 'date'
        elif col_lower in ['wte', 'water_table_elevation', 'lev_va']:
            col_mapping[col] = 'wte'

    well_ts = well_ts.rename(columns=col_mapping)

    # Convert date column
    if 'date' in well_ts.columns:
        well_ts['date'] = pd.to_datetime(well_ts['date'])

    print(f"Loaded groundwater time series: {len(well_ts):,} measurements")
    print(f"  Unique wells in time series: {well_ts['well_id'].nunique():,}")
    if 'date' in well_ts.columns:
        print(f"  Date range: {well_ts['date'].min()} to {well_ts['date'].max()}")

    return wells_gdf, well_ts, well_info


def load_water_level_timeseries(
    filepath: str,
    well_id_col: str = 'well_id',
    date_col: str = 'date',
    wte_col: str = 'wte'
) -> pd.DataFrame:
    """
    Load groundwater level time series from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file containing water level measurements.
    well_id_col : str, default 'well_id'
        Column name for well identifier
    date_col : str, default 'date'
        Column name for measurement date
    wte_col : str, default 'wte'
        Column name for water table elevation

    Returns
    -------
    pd.DataFrame
        DataFrame with well_id, date, wte columns

    Example
    -------
    >>> well_ts = load_water_level_timeseries('data/raw/groundwater/water_levels.csv')
    """
    df = pd.read_csv(filepath)

    # Standardize column names (handle case variations)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in [well_id_col.lower(), 'wellid', 'site_no']:
            col_mapping[col] = 'well_id'
        elif col_lower == date_col.lower():
            col_mapping[col] = 'date'
        elif col_lower in [wte_col.lower(), 'water_table_elevation', 'lev_va']:
            col_mapping[col] = 'wte'

    df = df.rename(columns=col_mapping)

    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded water level time series: {len(df):,} measurements")
    print(f"  Unique wells: {df['well_id'].nunique():,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def load_streamflow_data(
    filepath: str,
    gage_id_col: str = 'gage_id',
    date_col: str = 'date',
    q_col: str = 'q',
    filter_bfd: bool = True
) -> pd.DataFrame:
    """
    Load daily streamflow data from CSV file or directory of CSV files.

    Parameters
    ----------
    filepath : str
        Path to CSV file or directory containing CSV files.
        If directory, expects files named as gage_id.csv (e.g., '10011500.csv').
        Expected columns: date, Q (or q), ML_BFD (if filtering by baseflow)
    gage_id_col : str, default 'gage_id'
        Column name for gage identifier (only used for single file mode)
    date_col : str, default 'date'
        Column name for date
    q_col : str, default 'q'
        Column name for discharge
    filter_bfd : bool, default True
        If True and loading from directory, filter rows where ML_BFD == 1
        (baseflow-dominated conditions). Ignored for single file mode.

    Returns
    -------
    pd.DataFrame
        DataFrame with gage_id, date, q columns (and bfd if from directory)

    Example
    -------
    >>> streamflow = load_streamflow_data('data/raw/streamflow/daily_discharge.csv')
    >>> streamflow = load_streamflow_data('data/raw/streamflow/GSLB_ML')
    """
    # Check if filepath is a directory
    if os.path.isdir(filepath):
        # Load from directory (multiple CSV files)
        compiled_data = pd.DataFrame(columns=['gage_id', 'date', 'q'])
        
        # Iterate over each file in the directory
        for filename in os.listdir(filepath):
            if filename.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(filepath, filename)
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Filter rows where ML_BFD is 1 (if filter_bfd is True)
                if filter_bfd and 'ML_BFD' in df.columns:
                    df = df[df['ML_BFD'] == 1]
                
                # Extract gage_id from the filename (filename is the gage_id)
                gage_id = os.path.splitext(filename)[0]
                
                # Add a new column for gage_id
                df['gage_id'] = gage_id
                
                # Standardize column names
                col_mapping = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower == date_col.lower():
                        col_mapping[col] = 'date'
                    elif col_lower in [q_col.lower(), 'q', 'discharge', 'flow', 'streamflow']:
                        col_mapping[col] = 'q'
                    elif col_lower in ['ml_bfd', 'bfd']:
                        col_mapping[col] = 'bfd'
                
                df = df.rename(columns=col_mapping)
                
                # Select and rename necessary columns
                if 'bfd' in df.columns:
                    df = df[['gage_id', 'date', 'q', 'bfd']]
                else:
                    df = df[['gage_id', 'date', 'q']]
                
                # Append to the compiled DataFrame
                compiled_data = pd.concat([compiled_data, df], ignore_index=True)
        
        df = compiled_data
    else:
        # Load from single file
        df = pd.read_csv(filepath)

        # Standardize column names
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in [gage_id_col.lower(), 'site_no', 'station_id']:
                col_mapping[col] = 'gage_id'
            elif col_lower == date_col.lower():
                col_mapping[col] = 'date'
            elif col_lower in [q_col.lower(), 'q', 'discharge', 'flow', 'streamflow']:
                col_mapping[col] = 'q'

        df = df.rename(columns=col_mapping)
    
    # Ensure date column exists and convert to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError(f"Date column not found. Available columns: {df.columns.tolist()}")

    print(f"Loaded streamflow data: {len(df):,} records")
    print(f"  Gages: {df['gage_id'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def load_gage_info(filepath: str) -> pd.DataFrame:
    """
    Load stream gage location and metadata.

    Parameters
    ----------
    filepath : str
        Path to CSV file containing gage information.
        Expected columns: id, name, latitude, longitude

    Returns
    -------
    pd.DataFrame
        DataFrame with gage metadata

    Example
    -------
    >>> gage_info = load_gage_info('data/raw/streamflow/gage_info.csv')
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()

    # Standardize column names
    col_mapping = {
        'samplingfeaturecode': 'id',
        'site_no': 'id',
        'station_id': 'id',
        'lat': 'latitude',
        'lon': 'longitude',
        'lng': 'longitude',
    }
    df = df.rename(columns=col_mapping)

    print(f"Loaded gage information: {len(df)} gages")

    return df


def load_baseflow_classification(filepath: str) -> pd.DataFrame:
    """
    Load baseflow-dominated period classification.

    Parameters
    ----------
    filepath : str
        Path to CSV file with BFD classification.
        Expected columns: gage_id, date, bfd (0 or 1)

    Returns
    -------
    pd.DataFrame
        DataFrame with gage_id, date, bfd columns

    Example
    -------
    >>> bfd = load_baseflow_classification('data/raw/bfd/bfd_classification.csv')
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded BFD classification: {len(df):,} records")
    if 'bfd' in df.columns:
        print(f"  BFD=1: {(df['bfd'] == 1).sum():,} ({(df['bfd'] == 1).mean()*100:.1f}%)")

    return df


def load_reach_elevations(filepath: str) -> pd.DataFrame:
    """
    Load stream reach elevation data.

    Parameters
    ----------
    filepath : str
        Path to CSV file containing reach elevations.
        Expected columns: Reach_ID, Avg_GSE (average ground surface elevation in meters)

    Returns
    -------
    pd.DataFrame
        DataFrame with Reach_ID, Avg_GSE columns

    Example
    -------
    >>> reach_elev = load_reach_elevations('data/processed/reach_elevations.csv')
    """
    df = pd.read_csv(filepath)

    print(f"Loaded reach elevation data: {len(df)} reaches")
    if 'Avg_GSE' in df.columns:
        print(f"  Elevation range: {df['Avg_GSE'].min():.0f} - {df['Avg_GSE'].max():.0f} m")

    return df
