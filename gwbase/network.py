"""
Stream network analysis functions for GWBASE.

This module implements Step 1 of the GWBASE workflow:
- Build directed graph from stream network
- Identify terminal gages (downstream-most gages with no other gages downstream)
- Delineate upstream catchments for each terminal gage
"""

import pandas as pd
import geopandas as gpd
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional


def build_stream_network_graph(stream_gdf: gpd.GeoDataFrame) -> nx.DiGraph:
    """
    Build a directed graph representing the stream network.

    The graph uses catchment IDs (LINKNO) as nodes and downstream links
    (DSLINKNO) to define edges representing flow direction.

    Parameters
    ----------
    stream_gdf : gpd.GeoDataFrame
        Stream network GeoDataFrame with columns:
        - LINKNO: Unique identifier for each stream segment
        - DSLINKNO: Downstream link number (0 or NaN for outlets)

    Returns
    -------
    nx.DiGraph
        Directed graph where edges point downstream

    Example
    -------
    >>> G = build_stream_network_graph(stream_gdf)
    >>> print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    """
    G = nx.DiGraph()

    # Add edges based on stream connectivity
    for _, row in stream_gdf.iterrows():
        linkno = row['LINKNO']
        dslinkno = row['DSLINKNO']

        # Only add edge if valid downstream link exists
        if pd.notna(dslinkno) and dslinkno > 0:
            # Check that both LINKNO and DSLINKNO are valid
            if pd.notna(linkno) and linkno > 0:
                G.add_edge(int(linkno), int(dslinkno))

    print(f"Built stream network graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def match_gages_to_catchments(
    gage_df: pd.DataFrame,
    catchment_gdf: gpd.GeoDataFrame,
    linkno_col: str = 'linkno'
) -> pd.DataFrame:
    """
    Match stream gages to their containing catchments using spatial join.

    Parameters
    ----------
    gage_df : pd.DataFrame
        DataFrame with gage locations (longitude, latitude columns)
    catchment_gdf : gpd.GeoDataFrame
        GeoDataFrame with catchment polygons
    linkno_col : str, default 'linkno'
        Name of the column containing catchment IDs

    Returns
    -------
    pd.DataFrame
        DataFrame with gage-catchment matches

    Example
    -------
    >>> matched = match_gages_to_catchments(gage_df, catchment_gdf)
    """
    # Create GeoDataFrame from gage locations
    gage_gdf = gpd.GeoDataFrame(
        gage_df,
        geometry=gpd.points_from_xy(gage_df['longitude'], gage_df['latitude']),
        crs=catchment_gdf.crs
    )

    # Find the linkno column (case-insensitive)
    linkno_col_actual = None
    for col in catchment_gdf.columns:
        if col.lower() == linkno_col.lower():
            linkno_col_actual = col
            break

    if linkno_col_actual is None:
        raise ValueError(f"Column '{linkno_col}' not found in catchment GeoDataFrame")

    # Spatial join to find which catchment each gage falls within
    matched_gages = gpd.sjoin(
        gage_gdf[['id', 'name', 'geometry']],
        catchment_gdf[[linkno_col_actual, 'geometry']],
        how='inner',
        predicate='within'
    ).rename(columns={linkno_col_actual: 'catchment_id'})

    matched_gages = matched_gages[['id', 'name', 'geometry', 'catchment_id']]

    print(f"Matched {len(matched_gages)} gages to catchments")

    return matched_gages


def identify_terminal_gages(
    matched_gages: pd.DataFrame,
    stream_graph: nx.DiGraph,
    manual_remove: List[int] = None,
    manual_add: List[int] = None
) -> List[int]:
    """
    Identify terminal gages in the stream network.

    A terminal gage is defined as a gage that has no other gage located downstream.
    This prevents redundant processing of overlapping watersheds.

    Parameters
    ----------
    matched_gages : pd.DataFrame
        DataFrame from match_gages_to_catchments with 'id' and 'catchment_id'
    stream_graph : nx.DiGraph
        Directed graph of stream network from build_stream_network_graph
    manual_remove : list of int, optional
        Gage IDs to manually remove from terminal list
    manual_add : list of int, optional
        Gage IDs to manually add to terminal list

    Returns
    -------
    list of int
        List of terminal gage IDs

    Example
    -------
    >>> terminal_ids = identify_terminal_gages(matched_gages, G)
    >>> print(f"Found {len(terminal_ids)} terminal gages")
    """
    # Create mapping of gage IDs to their catchment IDs
    gage_links = dict(zip(matched_gages['id'], matched_gages['catchment_id']))

    terminal_ids = []

    # Check each gage to see if it's terminal
    for g1_id, g1_link in gage_links.items():
        is_terminal = True

        for g2_id, g2_link in gage_links.items():
            if g1_id != g2_id:
                try:
                    # If there's a path from g1 to g2, then g1 is not terminal
                    if nx.has_path(stream_graph, g1_link, g2_link):
                        is_terminal = False
                        break
                except nx.NetworkXError:
                    continue

        if is_terminal:
            terminal_ids.append(g1_id)

    # Apply manual adjustments
    if manual_remove:
        terminal_ids = [gid for gid in terminal_ids if gid not in manual_remove]

    if manual_add:
        for gid in manual_add:
            if gid in matched_gages['id'].values and gid not in terminal_ids:
                terminal_ids.append(gid)

    print(f"Identified {len(terminal_ids)} terminal gages")

    return terminal_ids


def get_upstream_catchments(
    terminal_gage_id: int,
    terminal_catchment_id: int,
    stream_graph: nx.DiGraph
) -> Set[int]:
    """
    Get all upstream catchment IDs that drain to a terminal gage.

    Parameters
    ----------
    terminal_gage_id : int
        ID of the terminal gage
    terminal_catchment_id : int
        Catchment ID where the terminal gage is located
    stream_graph : nx.DiGraph
        Directed graph of stream network

    Returns
    -------
    set of int
        Set of catchment IDs upstream of the terminal gage

    Example
    -------
    >>> upstream = get_upstream_catchments(10126000, 123456, G)
    >>> print(f"Found {len(upstream)} upstream catchments")
    """
    upstream_ids = set()

    # Check each node in the graph
    for node in stream_graph.nodes:
        # If there's a path from this node to the terminal catchment,
        # then this node is upstream
        if nx.has_path(stream_graph, node, terminal_catchment_id):
            upstream_ids.add(node)

    # Include the terminal gage's own catchment
    upstream_ids.add(terminal_catchment_id)

    return upstream_ids


def delineate_all_upstream_catchments(
    terminal_gages: pd.DataFrame,
    stream_graph: nx.DiGraph
) -> pd.DataFrame:
    """
    Delineate upstream catchments for all terminal gages.

    Parameters
    ----------
    terminal_gages : pd.DataFrame
        DataFrame with terminal gage info including 'id', 'name', 'catchment_id'
    stream_graph : nx.DiGraph
        Directed graph of stream network

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Gage_ID: Terminal gage ID
        - Gage_Name: Terminal gage name
        - Terminal_Catchment_ID: Catchment ID at the gage
        - Upstream_Catchment_ID: ID of each upstream catchment

    Example
    -------
    >>> upstream_df = delineate_all_upstream_catchments(terminal_gages_df, G)
    >>> upstream_df.to_csv('data/processed/terminal_gage_upstream_catchments.csv')
    """
    records = []

    for _, gage in terminal_gages.iterrows():
        gage_id = gage['id']
        gage_name = gage.get('name', f'Gage {gage_id}')
        catchment_id = gage['catchment_id']

        # Get all upstream catchments
        upstream_ids = get_upstream_catchments(gage_id, catchment_id, stream_graph)

        # Create records for each upstream catchment
        for up_id in upstream_ids:
            records.append({
                'Gage_ID': gage_id,
                'Gage_Name': gage_name,
                'Terminal_Catchment_ID': catchment_id,
                'Upstream_Catchment_ID': up_id
            })

    df_upstream = pd.DataFrame(records)

    print(f"Delineated upstream catchments for {df_upstream['Gage_ID'].nunique()} terminal gages")
    print(f"Total upstream catchment records: {len(df_upstream)}")

    return df_upstream


def find_downstream_gage(
    reach_id: int,
    stream_gdf: gpd.GeoDataFrame,
    gages_df: pd.DataFrame,
    max_path_length: int = 100
) -> Optional[int]:
    """
    Find the downstream gage for a given reach using network topology.

    Parameters
    ----------
    reach_id : int
        Starting reach ID (LINKNO)
    stream_gdf : gpd.GeoDataFrame
        Stream network with LINKNO and DSLINKNO columns
    gages_df : pd.DataFrame
        Gage data with COMID_v2 column for reach matching
    max_path_length : int, default 100
        Maximum number of reaches to traverse downstream

    Returns
    -------
    int or None
        Downstream gage ID, or None if not found

    Example
    -------
    >>> downstream_gage = find_downstream_gage(123456, stream_gdf, gages_df)
    """
    current_reach = reach_id
    visited_reaches = set()
    path_length = 0

    while current_reach and path_length < max_path_length:
        if current_reach in visited_reaches:
            break
        visited_reaches.add(current_reach)

        # Check for gage on current reach (if COMID_v2 column exists)
        if 'COMID_v2' in gages_df.columns:
            gage_on_reach = gages_df[gages_df['COMID_v2'] == current_reach]
            if not gage_on_reach.empty:
                # Return the gage ID (using samplingFeatureCode if available)
                if 'samplingFeatureCode' in gage_on_reach.columns:
                    return gage_on_reach.iloc[0]['samplingFeatureCode']
                elif 'id' in gage_on_reach.columns:
                    return gage_on_reach.iloc[0]['id']

        # Move downstream
        reach_row = stream_gdf[stream_gdf['LINKNO'] == current_reach]
        if reach_row.empty:
            break

        downstream_id = reach_row.iloc[0]['DSLINKNO']
        if downstream_id == 0 or pd.isna(downstream_id):
            break

        current_reach = downstream_id
        path_length += 1

    return None
