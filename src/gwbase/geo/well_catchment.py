"""Well-catchment association functions."""
import pandas as pd
import geopandas as gpd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def assign_wells_to_upstream_catchments(
    wells_gdf: gpd.GeoDataFrame,
    terminal_upstream_df: pd.DataFrame,
    subbasin_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Assign wells to terminal gages based on catchment membership.
    
    A well is assigned to a terminal gage if it falls within any of that
    gage's upstream catchments (including the terminal catchment itself).
    
    Parameters:
    -----------
    wells_gdf : GeoDataFrame
        Wells with geometry column
    terminal_upstream_df : DataFrame
        Output from Step 1 with columns: Gage_ID, Gage_Name, 
        Terminal_Catchment_ID, Upstream_Catchment_ID
    subbasin_gdf : GeoDataFrame
        Catchment polygons with linkno/LINKNO column
        
    Returns:
    --------
    DataFrame
        Columns: well_id, gage_id, gage_name, catchment_id
    """
    logger.info("Assigning wells to upstream catchments")
    
    # Find linkno column in subbasin
    linkno_col = None
    for col in ['linkno', 'LINKNO', 'LINK_NO']:
        if col in subbasin_gdf.columns:
            linkno_col = col
            break
    
    if linkno_col is None:
        raise ValueError("Subbasin GeoDataFrame must have linkno/LINKNO column")
    
    # Ensure CRS match
    if wells_gdf.crs != subbasin_gdf.crs:
        logger.info(f"Converting wells CRS from {wells_gdf.crs} to {subbasin_gdf.crs}")
        wells_gdf = wells_gdf.to_crs(subbasin_gdf.crs)
    
    # Spatial join: find which catchment each well is in
    wells_with_catchments = wells_gdf.sjoin(
        subbasin_gdf[[linkno_col, 'geometry']],
        how='inner',
        predicate='within'
    ).rename(columns={linkno_col: 'catchment_id'})
    
    # Create mapping from catchment_id to terminal gage(s)
    catchment_to_gages = {}
    for _, row in terminal_upstream_df.iterrows():
        catchment = row['Upstream_Catchment_ID']
        gage_id = row['Gage_ID']
        gage_name = row['Gage_Name']
        
        if catchment not in catchment_to_gages:
            catchment_to_gages[catchment] = []
        catchment_to_gages[catchment].append({
            'gage_id': gage_id,
            'gage_name': gage_name
        })
    
    # Find well_id column
    well_id_col = None
    for col in ['well_id', 'WELL_ID', 'Well_ID', 'wellid']:
        if col in wells_with_catchments.columns:
            well_id_col = col
            break
    
    if well_id_col is None:
        raise ValueError("Cannot find well_id column in wells_with_catchments")
    
    # Assign wells to gages
    records = []
    for _, well_row in wells_with_catchments.iterrows():
        well_id = well_row[well_id_col]
        catchment_id = well_row['catchment_id']
        
        if catchment_id in catchment_to_gages:
            for gage_info in catchment_to_gages[catchment_id]:
                records.append({
                    'well_id': well_id,
                    'gage_id': gage_info['gage_id'],
                    'gage_name': gage_info['gage_name'],
                    'catchment_id': catchment_id
                })
        else:
            logger.warning(f"Well {well_id} in catchment {catchment_id} not assigned to any terminal gage")
    
    result = pd.DataFrame(records)
    logger.info(f"Assigned {result['well_id'].nunique()} wells to "
                f"{result['gage_id'].nunique()} terminal gages")
    
    return result

