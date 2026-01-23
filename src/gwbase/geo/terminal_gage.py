"""Terminal gage identification."""
import pandas as pd
import geopandas as gpd
import networkx as nx
from typing import List, Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)


def identify_terminal_gages(
    gages_gdf: gpd.GeoDataFrame,
    stream_gdf: gpd.GeoDataFrame,
    subbasin_gdf: gpd.GeoDataFrame,
    manual_remove: Optional[List[int]] = None,
    manual_add: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Identify terminal (downstream-most) gages.
    
    A terminal gage has NO OTHER GAGE downstream in the directed network.
    
    Parameters:
    -----------
    gages_gdf : GeoDataFrame
        Gages with 'id' and geometry columns
    stream_gdf : GeoDataFrame
        Stream network with 'LINKNO' and 'DSLINKNO' columns
    subbasin_gdf : GeoDataFrame
        Catchment polygons (for spatial join)
    manual_remove : list of int, optional
        Gage IDs to manually remove from terminal list
    manual_add : list of int, optional
        Gage IDs to manually add to terminal list
        
    Returns:
    --------
    DataFrame
        Columns: Gage_ID, Gage_Name, Terminal_Catchment_ID, Upstream_Catchment_ID
    """
    logger.info("Building stream network graph")
    G = nx.DiGraph()
    
    # Build directed graph: LINKNO -> DSLINKNO (downstream direction)
    linkno_col = None
    dslinkno_col = None
    
    for col in ['LINKNO', 'linkno', 'LINK_NO']:
        if col in stream_gdf.columns:
            linkno_col = col
            break
    
    for col in ['DSLINKNO', 'dslinkno', 'DS_LINKNO']:
        if col in stream_gdf.columns:
            dslinkno_col = col
            break
    
    if linkno_col is None or dslinkno_col is None:
        raise ValueError(f"Stream network must have LINKNO and DSLINKNO columns")
    
    for _, row in stream_gdf.iterrows():
        if (pd.notna(row[linkno_col]) and pd.notna(row[dslinkno_col]) 
            and row[dslinkno_col] > 0):
            G.add_edge(int(row[linkno_col]), int(row[dslinkno_col]))
    
    # Spatial join: assign catchments to gages
    linkno_subbasin = None
    for col in ['linkno', 'LINKNO', 'LINK_NO']:
        if col in subbasin_gdf.columns:
            linkno_subbasin = col
            break
    
    if linkno_subbasin is None:
        raise ValueError("Subbasin GeoDataFrame must have linkno/LINKNO column")
    
    # Perform spatial join
    matched_gages = gages_gdf.sjoin(
        subbasin_gdf[[linkno_subbasin, 'geometry']],
        how='inner',
        predicate='within'
    ).rename(columns={linkno_subbasin: 'catchment_id'})
    
    # Create gage to catchment mapping
    gage_links = dict(zip(matched_gages['id'], matched_gages['catchment_id']))
    terminal_ids = []
    
    logger.info("Identifying terminal gages")
    # A terminal gage has no other gage downstream
    for g1_id, g1_catchment in gage_links.items():
        is_terminal = True
        
        # Check if any OTHER gage is downstream of g1
        for g2_id, g2_catchment in gage_links.items():
            if g1_id == g2_id:
                continue
            
            try:
                if nx.has_path(G, g1_catchment, g2_catchment):
                    is_terminal = False
                    break
            except (KeyError, ValueError):
                continue
        
        if is_terminal:
            terminal_ids.append(g1_id)
    
    # Apply manual adjustments
    if manual_remove:
        logger.info(f"Manually removing {len(manual_remove)} gages")
        terminal_ids = [gid for gid in terminal_ids if gid not in manual_remove]
    
    if manual_add:
        logger.info(f"Manually adding {len(manual_add)} gages")
        for gage_id in manual_add:
            if gage_id in matched_gages['id'].values and gage_id not in terminal_ids:
                terminal_ids.append(gage_id)
    
    logger.info(f"Found {len(terminal_ids)} terminal gages")
    
    # Get terminal gages DataFrame
    terminal_gages = matched_gages[matched_gages['id'].isin(terminal_ids)].copy()
    
    # Find all upstream catchments for each terminal gage
    logger.info("Mapping upstream catchments")
    records = []
    for _, gage in terminal_gages.iterrows():
        upstream_ids: Set[int] = set()
        gage_catchment = gage['catchment_id']
        
        # Find all nodes that have a path TO this terminal gage's catchment
        for node in G.nodes():
            try:
                if nx.has_path(G, node, gage_catchment):
                    upstream_ids.add(node)
            except (KeyError, ValueError):
                continue
        
        upstream_ids.add(gage_catchment)
        
        # Create records
        gage_name = gage.get('name', f"Gage_{gage['id']}")
        for up_id in upstream_ids:
            records.append({
                'Gage_ID': gage['id'],
                'Gage_Name': gage_name,
                'Terminal_Catchment_ID': gage_catchment,
                'Upstream_Catchment_ID': up_id
            })
    
    result = pd.DataFrame(records)
    logger.info(f"Mapped {result['Upstream_Catchment_ID'].nunique()} catchments "
                f"to {result['Gage_ID'].nunique()} terminal gages")
    
    return result
