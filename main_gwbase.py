#!/usr/bin/env python3
"""
GWBASE Main Workflow Script

This script runs the complete GWBASE workflow for assessing the impact
of groundwater decline on baseflow in streams.

The workflow consists of 9 steps:
1. Identify Stream Network and Upstream Catchments
2. Locate Groundwater Wells within Catchments
3. Associate Wells with Nearest Stream Segments
4. Filter Wells with Insufficient Data
5. Temporal Interpolation of Groundwater Levels (PCHIP)
6. Elevation-Based Filtering
7. Pair Groundwater and Streamflow Records under Baseflow-Dominated Conditions
8. Compute ΔWTE and ΔQ
9. Analyze ΔWTE–ΔQ Relationships

Usage:
    python main_gwbase.py --config config.yaml
    python main_gwbase.py --data-dir data/

Author: GWBASE Development Team
"""

import argparse
import os
from pathlib import Path
import pandas as pd

# Import GWBASE modules
import gwbase


def setup_directories(base_dir: str, output_dir: str = None) -> dict:
    """Set up directory structure for GWBASE analysis."""
    from datetime import datetime
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, 'reports', f'output_{timestamp}')
    else:
        # Use provided output directory (can be relative or absolute)
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(base_dir, output_dir)
    
    dirs = {
        'raw': os.path.join(base_dir, 'data', 'raw'),
        'processed': os.path.join(output_dir, 'processed'),
        'features': os.path.join(output_dir, 'features'),
        'figures': os.path.join(output_dir, 'figures'),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Output directory: {output_dir}")

    return dirs


def run_step_1_network_analysis(
    stream_gdf,
    catchment_gdf,
    gage_df,
    output_dir: str
):
    """
    Step 1: Identify Stream Network and Upstream Catchments

    - Build directed graph from stream network
    - Match gages to catchments
    - Identify terminal gages
    - Delineate upstream catchments
    """
    print("\n" + "="*60)
    print("STEP 1: Stream Network and Upstream Catchments")
    print("="*60)

    # Build stream network graph
    G = gwbase.build_stream_network_graph(stream_gdf)

    # Match gages to catchments
    matched_gages = gwbase.match_gages_to_catchments(gage_df, catchment_gdf)

    # Identify terminal gages
    terminal_ids = gwbase.identify_terminal_gages(matched_gages, G)

    # Get terminal gage info
    terminal_gages = matched_gages[matched_gages['id'].isin(terminal_ids)]

    # Delineate upstream catchments
    upstream_df = gwbase.delineate_all_upstream_catchments(terminal_gages, G)

    # Save results
    upstream_df.to_csv(os.path.join(output_dir, 'terminal_gage_upstream_catchments.csv'), index=False)
    terminal_gages.to_csv(os.path.join(output_dir, 'terminal_gages.csv'), index=False)

    print(f"\nStep 1 complete. Identified {len(terminal_ids)} terminal gages.")

    return G, matched_gages, terminal_gages, upstream_df


def run_step_2_locate_wells(
    wells_gdf,
    catchment_gdf,
    upstream_df,
    output_dir: str
):
    """
    Step 2: Locate Groundwater Wells within Catchments

    - Spatial join wells to catchments
    - Assign wells to terminal gages
    """
    print("\n" + "="*60)
    print("STEP 2: Locate Wells within Catchments")
    print("="*60)

    # Locate wells in catchments
    wells_with_gages = gwbase.locate_wells_in_catchments(
        wells_gdf, catchment_gdf, upstream_df
    )

    # Save results
    wells_with_gages.to_csv(os.path.join(output_dir, 'wells_in_catchments.csv'), index=False)

    print(f"\nStep 2 complete. Located {wells_with_gages['well_id'].nunique()} wells.")

    return wells_with_gages


def run_step_3_associate_reaches(
    wells_gdf,
    stream_gdf,
    gages_df,
    reach_elevations,
    output_dir: str
):
    """
    Step 3: Associate Wells with Nearest Stream Segments

    - Find nearest reach for each well
    - Get reach elevation
    - Find downstream gage
    """
    print("\n" + "="*60)
    print("STEP 3: Associate Wells with Reaches")
    print("="*60)

    # Associate wells with reaches
    well_reach = gwbase.associate_wells_with_reaches(
        wells_gdf, stream_gdf, gages_df, reach_elevations
    )

    # Save results
    well_reach.to_csv(os.path.join(output_dir, 'well_reach_relationships.csv'), index=False)

    print(f"\nStep 3 complete.")

    return well_reach


def run_step_4_preprocessing(
    well_ts: pd.DataFrame,
    output_dir: str,
    min_points: int = 5
):
    """
    Step 4: Filter Wells with Insufficient Data

    - Detect and remove outliers
    - Filter wells based on data quality
    """
    print("\n" + "="*60)
    print("STEP 4: Data Preprocessing")
    print("="*60)

    # Clean well data
    clean_data = gwbase.clean_well_data_for_interpolation(well_ts, min_points=min_points)

    # Save results
    clean_data.to_csv(os.path.join(output_dir, 'well_ts_cleaned.csv'), index=False)

    print(f"\nStep 4 complete. {clean_data['well_id'].nunique()} wells ready for interpolation.")

    return clean_data


def run_step_5_interpolation(
    clean_data: pd.DataFrame,
    well_info: pd.DataFrame,
    output_dir: str
):
    """
    Step 5: Temporal Interpolation of Groundwater Levels

    - PCHIP interpolation to daily resolution
    - Merge with well location info
    """
    print("\n" + "="*60)
    print("STEP 5: PCHIP Interpolation")
    print("="*60)

    # Interpolate
    daily_data = gwbase.interpolate_with_well_info(clean_data, well_info)

    # Save results
    daily_data.to_csv(os.path.join(output_dir, 'well_pchip_daily.csv'), index=False)

    print(f"\nStep 5 complete. Generated {len(daily_data):,} daily records.")

    return daily_data


def run_step_6_elevation_filtering(
    daily_data: pd.DataFrame,
    well_reach: pd.DataFrame,
    output_dir: str,
    buffer_m: float = 30.0,
    wells_with_gages: pd.DataFrame = None
):
    """
    Step 6: Elevation-Based Filtering

    - Filter wells based on WTE vs stream elevation
    - Retain wells with potential stream connectivity
    """
    print("\n" + "="*60)
    print("STEP 6: Elevation-Based Filtering")
    print("="*60)

    # Filter by elevation
    filtered_data, dist_stats = gwbase.filter_by_elevation(
        daily_data, well_reach, distance_buffer_meters=buffer_m
    )

    # Add gage_id before saving (needed for Step 7)
    if 'gage_id' not in filtered_data.columns:
        print("\nAdding gage_id to filtered_data...")
        # Primary source: well_reach (Downstream_Gage)
        if well_reach is not None and 'Downstream_Gage' in well_reach.columns:
            well_reach_mapping = well_reach[['Well_ID', 'Downstream_Gage']].rename(columns={
                'Well_ID': 'well_id',
                'Downstream_Gage': 'gage_id'
            }).dropna(subset=['gage_id']).drop_duplicates()
            filtered_data = pd.merge(
                filtered_data,
                well_reach_mapping,
                on='well_id',
                how='left'
            )
            print(f"  Added {filtered_data['gage_id'].notna().sum():,} gage_ids from well_reach")
        
        # Secondary source: wells_with_gages (fill missing)
        if wells_with_gages is not None and 'gage_id' in wells_with_gages.columns:
            gage_mapping = wells_with_gages[['well_id', 'gage_id']].drop_duplicates()
            missing_mask = filtered_data['gage_id'].isna()
            if missing_mask.any():
                temp_merge = pd.merge(
                    filtered_data[missing_mask][['well_id']],
                    gage_mapping,
                    on='well_id',
                    how='left'
                )
                filtered_data.loc[missing_mask, 'gage_id'] = temp_merge['gage_id'].values
                print(f"  Filled {temp_merge['gage_id'].notna().sum():,} additional gage_ids from wells_with_gages")
        
        # Convert to string
        filtered_data['gage_id'] = filtered_data['gage_id'].astype(str)
        filtered_data['gage_id'] = filtered_data['gage_id'].replace('nan', pd.NA)

    # Save results
    filtered_data.to_csv(os.path.join(output_dir, 'filtered_by_elevation.csv'), index=False)
    dist_stats.to_csv(os.path.join(output_dir, 'elevation_distribution.csv'), index=False)

    print(f"\nStep 6 complete. {len(filtered_data):,} records retained.")

    return filtered_data


def run_step_7_pairing(
    well_data: pd.DataFrame,
    streamflow_data: pd.DataFrame,
    bfd_classification: pd.DataFrame = None,
    output_dir: str = None
):
    """
    Step 7: Pair Groundwater and Streamflow Records

    - Match well and streamflow by gage and date
    - Add BFD classification (from streamflow data if available, or from bfd_classification)
    - Calculate baseline values
    """
    print("\n" + "="*60)
    print("STEP 7: Well-Streamflow Pairing")
    print("="*60)

    # Pair data
    paired = gwbase.pair_wells_with_streamflow(
        well_data, streamflow_data, bfd_classification
    )

    # Calculate baseline values
    paired = gwbase.calculate_baseline_values(paired)

    # Save results
    paired.to_csv(os.path.join(output_dir, 'paired_well_streamflow.csv'), index=False)

    print(f"\nStep 7 complete. {len(paired):,} paired records.")

    return paired


def run_step_8_delta_metrics(
    paired_data: pd.DataFrame,
    output_dir: str
):
    """
    Step 8: Compute ΔWTE and ΔQ

    - Calculate change from baseline
    - Create lagged versions
    """
    print("\n" + "="*60)
    print("STEP 8: Delta Metrics Computation")
    print("="*60)

    # Compute delta metrics
    data_with_deltas = gwbase.compute_delta_metrics(paired_data)

    # Save results
    data_with_deltas.to_csv(os.path.join(output_dir, 'data_with_deltas.csv'), index=False)

    # Create lag versions
    for lag, unit in [(1, 'years'), (3, 'months'), (6, 'months')]:
        lag_data = gwbase.create_lag_analysis(data_with_deltas, lag, unit)
        suffix = f'{lag}yr' if unit == 'years' else f'{lag}mo'
        lag_data.to_csv(os.path.join(output_dir, f'data_lag_{suffix}.csv'), index=False)

    print(f"\nStep 8 complete.")

    return data_with_deltas


def run_step_9_analysis(
    data_with_deltas: pd.DataFrame,
    output_dir: str,
    figures_dir: str
):
    """
    Step 9: Analyze ΔWTE–ΔQ Relationships

    - Linear regression by gage
    - Linear regression by well
    - Mutual information analysis
    - Cross-correlation analysis
    """
    print("\n" + "="*60)
    print("STEP 9: Regression and Correlation Analysis")
    print("="*60)

    # Regression by gage
    gage_stats = gwbase.compute_regression_by_gage(data_with_deltas)
    gage_stats.to_csv(os.path.join(output_dir, 'regression_by_gage.csv'), index=False)

    # Regression by well
    well_stats = gwbase.compute_regression_by_well(data_with_deltas)
    well_stats.to_csv(os.path.join(output_dir, 'regression_by_well.csv'), index=False)

    # Summary
    summary = gwbase.summarize_regression_results(gage_stats, well_stats)

    # Mutual information analysis
    mi_results = gwbase.compute_mi_analysis(data_with_deltas)
    mi_results.to_csv(os.path.join(output_dir, 'mi_analysis.csv'), index=False)

    # Create visualizations
    gwbase.plot_regression_summary(gage_stats, os.path.join(figures_dir, 'regression'))
    gwbase.plot_delta_scatter(data_with_deltas, os.path.join(figures_dir, 'scatter_plots'))

    print(f"\nStep 9 complete.")

    return gage_stats, well_stats, mi_results


def main():
    """Main entry point for GWBASE workflow."""
    parser = argparse.ArgumentParser(
        description='GWBASE - Groundwater-Baseflow Analysis System'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='.',
        help='Base directory for data (default: current directory)'
    )
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Steps to run (e.g., "1-5" or "4,5,6" or "all")'
    )
    parser.add_argument(
        '--buffer',
        type=float,
        default=30.0,
        help='Elevation buffer in meters for Step 6 (default: 30)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory path (if not specified, creates new timestamped directory)'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("GWBASE - Groundwater-Baseflow Analysis System")
    print(f"Version: {gwbase.__version__}")
    print("="*60)

    # Setup directories
    dirs = setup_directories(args.data_dir, args.output_dir)

    print(f"\nData directory: {args.data_dir}")
    print(f"Processing steps: {args.steps}")

    # Parse steps to run
    start_step = 1
    end_step = 9
    if args.steps != 'all':
        if '-' in args.steps:
            # Range format: "1-5" or "6-9"
            parts = args.steps.split('-')
            start_step = int(parts[0])
            end_step = int(parts[1]) if len(parts) > 1 else 9
        elif ',' in args.steps:
            # Comma-separated: "4,5,6"
            steps_to_run = [int(s.strip()) for s in args.steps.split(',')]
            start_step = min(steps_to_run)
            end_step = max(steps_to_run)
        else:
            # Single step: "6"
            start_step = int(args.steps)
            end_step = int(args.steps)

    print(f"Running steps {start_step} to {end_step}")

    print("\n" + "-"*60)
    print("GWBASE workflow ready.")
    print("-"*60)

    # Load data (always needed)
    stream_gdf = gwbase.load_hydrography_data(
        os.path.join(dirs['raw'], 'hydrography/gslb_stream.shp')
    )
    catchment_gdf = gwbase.load_hydrography_data(
        os.path.join(dirs['raw'], 'hydrography/gsl_catchment.shp')
    )
    gage_df = gwbase.load_gage_info(
        os.path.join(dirs['raw'], 'streamflow/gsl_nwm_gage.csv')
    )
    # Load gsl_nwm.csv to get COMID_v2 information for reach matching
    try:
        gsl_nwm_df = pd.read_csv(os.path.join(dirs['raw'], 'streamflow/gsl_nwm.csv'))
        # Merge COMID_v2 from gsl_nwm.csv to gage_df
        if 'COMID_v2' in gsl_nwm_df.columns and 'id' in gage_df.columns:
            # Match by samplingFeatureCode (which contains the gage ID as string)
            if 'samplingFeatureCode' in gsl_nwm_df.columns:
                # Create a temporary column in gage_df for matching
                gage_df_temp = gage_df.copy()
                gage_df_temp['id_str'] = gage_df_temp['id'].astype(str)
                gsl_nwm_df_temp = gsl_nwm_df[['samplingFeatureCode', 'COMID_v2']].copy()
                gsl_nwm_df_temp['samplingFeatureCode_str'] = gsl_nwm_df_temp['samplingFeatureCode'].astype(str)
                # Merge
                gage_df = pd.merge(
                    gage_df_temp,
                    gsl_nwm_df_temp[['samplingFeatureCode_str', 'COMID_v2']],
                    left_on='id_str',
                    right_on='samplingFeatureCode_str',
                    how='left'
                )
                # Drop temporary columns
                gage_df = gage_df.drop(['id_str', 'samplingFeatureCode_str'], axis=1, errors='ignore')
            print(f"Merged COMID_v2 information: {gage_df['COMID_v2'].notna().sum()} gages have COMID_v2")
    except Exception as e:
        print(f"Warning: Could not load COMID_v2 from gsl_nwm.csv: {e}")
        print("Continuing without COMID_v2 (downstream gage finding may be limited)")
    
    wells_gdf, well_ts, well_info = gwbase.load_groundwater_data(
        well_locations_path=os.path.join(dirs['raw'], 'groundwater/GSLB_1900-2023_wells_with_aquifers.csv'),
        timeseries_path=os.path.join(dirs['raw'], 'groundwater/GSLB_1900-2023_TS_with_aquifers.csv')
    )
    streamflow = gwbase.load_streamflow_data(
        os.path.join(dirs['raw'], 'streamflow/GSLB_ML')
    )
    # Note: bfd_classification is optional - streamflow data already contains 'bfd' column
    # bfd_class = gwbase.load_baseflow_classification(
    #     os.path.join(dirs['raw'], 'bfd/bfd_classification.csv')
    # )
    reach_elev = gwbase.load_reach_elevations(
        os.path.join(dirs['raw'], 'streamflow/reach_centroids_with_Elev.csv')
    )

    # Run workflow with step selection
    # Initialize variables for intermediate results
    G, matched_gages, terminal_gages, upstream_df = None, None, None, None
    wells_with_gages = None
    well_reach = None
    clean_data = None
    daily_data = None
    filtered_data = None
    paired = None
    data_with_deltas = None

    # Step 1: Network Analysis
    if start_step <= 1 <= end_step:
        G, matched_gages, terminal_gages, upstream_df = run_step_1_network_analysis(
            stream_gdf, catchment_gdf, gage_df, dirs['processed']
        )
    elif start_step > 1:
        # Load previous results
        print(f"\nLoading Step 1 results from previous run...")
        upstream_df = pd.read_csv(os.path.join(dirs['processed'], 'terminal_gage_upstream_catchments.csv'))
        terminal_gages = pd.read_csv(os.path.join(dirs['processed'], 'terminal_gages.csv'))
        # Rebuild graph if needed for later steps
        G = gwbase.build_stream_network_graph(stream_gdf)
        matched_gages = gwbase.match_gages_to_catchments(gage_df, catchment_gdf)

    # Step 2: Locate Wells
    if start_step <= 2 <= end_step:
        wells_with_gages = run_step_2_locate_wells(
            wells_gdf, catchment_gdf, upstream_df, dirs['processed']
        )
    elif start_step > 2:
        print(f"\nLoading Step 2 results from previous run...")
        wells_with_gages = pd.read_csv(os.path.join(dirs['processed'], 'wells_in_catchments.csv'))

    # Step 3: Associate Reaches
    if start_step <= 3 <= end_step:
        well_reach = run_step_3_associate_reaches(
            wells_gdf, stream_gdf, gage_df, reach_elev, dirs['processed']
        )
    elif start_step > 3:
        print(f"\nLoading Step 3 results from previous run...")
        well_reach = pd.read_csv(os.path.join(dirs['processed'], 'well_reach_relationships.csv'))

    # Step 4: Preprocessing
    if start_step <= 4 <= end_step:
        clean_data = run_step_4_preprocessing(well_ts, dirs['processed'])
    elif start_step > 4:
        print(f"\nLoading Step 4 results from previous run...")
        clean_data = pd.read_csv(os.path.join(dirs['processed'], 'well_ts_cleaned.csv'))

    # Step 5: Interpolation
    if start_step <= 5 <= end_step:
        daily_data = run_step_5_interpolation(clean_data, well_info, dirs['processed'])
    elif start_step > 5:
        print(f"\nLoading Step 5 results from previous run...")
        daily_data = pd.read_csv(os.path.join(dirs['processed'], 'well_pchip_daily.csv'))
        daily_data['date'] = pd.to_datetime(daily_data['date'])

    # Step 6: Elevation Filtering
    if start_step <= 6 <= end_step:
        filtered_data = run_step_6_elevation_filtering(
            daily_data, well_reach, dirs['processed'], args.buffer, wells_with_gages
        )
    elif start_step > 6:
        print(f"\nLoading Step 6 results from previous run...")
        filtered_data = pd.read_csv(os.path.join(dirs['processed'], 'filtered_by_elevation.csv'))
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        # Add gage_id if missing
        if 'gage_id' not in filtered_data.columns or filtered_data['gage_id'].isna().all():
            print("\nAdding gage_id to filtered_data...")
            # Primary source: well_reach (Downstream_Gage) - has more coverage
            if well_reach is not None and 'Downstream_Gage' in well_reach.columns:
                well_reach_mapping = well_reach[['Well_ID', 'Downstream_Gage']].rename(columns={
                    'Well_ID': 'well_id',
                    'Downstream_Gage': 'gage_id'
                }).dropna(subset=['gage_id']).drop_duplicates()
                filtered_data = pd.merge(
                    filtered_data,
                    well_reach_mapping,
                    on='well_id',
                    how='left'
                )
                print(f"  Added {filtered_data['gage_id'].notna().sum():,} gage_ids from well_reach (Downstream_Gage)")
            
            # Secondary source: wells_with_gages (fill missing)
            if wells_with_gages is not None and 'gage_id' in wells_with_gages.columns:
                gage_mapping = wells_with_gages[['well_id', 'gage_id']].drop_duplicates()
                # Only fill missing gage_ids
                if 'gage_id' in filtered_data.columns:
                    missing_mask = filtered_data['gage_id'].isna()
                    if missing_mask.any():
                        temp_merge = pd.merge(
                            filtered_data[missing_mask][['well_id']],
                            gage_mapping,
                            on='well_id',
                            how='left'
                        )
                        filtered_data.loc[missing_mask, 'gage_id'] = temp_merge['gage_id'].values
                        print(f"  Filled {filtered_data['gage_id'].notna().sum() - (missing_mask.sum() - temp_merge['gage_id'].notna().sum()):,} additional gage_ids from wells_with_gages")
                else:
                    filtered_data = pd.merge(
                        filtered_data,
                        gage_mapping,
                        on='well_id',
                        how='left'
                    )
                    print(f"  Added {filtered_data['gage_id'].notna().sum():,} gage_ids from wells_with_gages")
            
            # Convert gage_id to string to match streamflow data type
            filtered_data['gage_id'] = filtered_data['gage_id'].astype(str)
            filtered_data['gage_id'] = filtered_data['gage_id'].replace('nan', pd.NA)
            
            # Re-save with gage_id
            filtered_data.to_csv(os.path.join(dirs['processed'], 'filtered_by_elevation.csv'), index=False)
    elif start_step > 6:
        print(f"\nLoading Step 6 results from previous run...")
        filtered_data = pd.read_csv(os.path.join(dirs['processed'], 'filtered_by_elevation.csv'))
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        # Add gage_id if missing
        if 'gage_id' not in filtered_data.columns:
            print("Adding gage_id to filtered_data...")
            if wells_with_gages is not None and 'gage_id' in wells_with_gages.columns:
                gage_mapping = wells_with_gages[['well_id', 'gage_id']].drop_duplicates()
                filtered_data = pd.merge(
                    filtered_data,
                    gage_mapping,
                    on='well_id',
                    how='left'
                )
            if well_reach is not None and 'Downstream_Gage' in well_reach.columns:
                well_reach_mapping = well_reach[['Well_ID', 'Downstream_Gage']].rename(columns={
                    'Well_ID': 'well_id',
                    'Downstream_Gage': 'gage_id_reach'
                }).dropna(subset=['gage_id_reach']).drop_duplicates()
                filtered_data = pd.merge(
                    filtered_data,
                    well_reach_mapping,
                    on='well_id',
                    how='left'
                )
                if 'gage_id' in filtered_data.columns:
                    filtered_data['gage_id'] = filtered_data['gage_id'].fillna(filtered_data['gage_id_reach'])
                else:
                    filtered_data['gage_id'] = filtered_data['gage_id_reach']
                filtered_data = filtered_data.drop('gage_id_reach', axis=1, errors='ignore')
            filtered_data['gage_id'] = filtered_data['gage_id'].astype(str)
            filtered_data['gage_id'] = filtered_data['gage_id'].replace('nan', pd.NA)
    
    # Ensure gage_id is string type in both filtered_data and streamflow for merging
    if filtered_data is not None and 'gage_id' in filtered_data.columns:
        # Convert float gage_id to string (remove .0 suffix)
        filtered_data = filtered_data.copy()
        filtered_data['gage_id'] = filtered_data['gage_id'].astype(str).str.replace('.0', '', regex=False)
        filtered_data['gage_id'] = filtered_data['gage_id'].replace('nan', pd.NA)
    
    if streamflow is not None and 'gage_id' in streamflow.columns:
        streamflow = streamflow.copy()
        streamflow['gage_id'] = streamflow['gage_id'].astype(str)

    # Step 7: Pairing
    if start_step <= 7 <= end_step:
        paired = run_step_7_pairing(filtered_data, streamflow, None, dirs['processed'])
    elif start_step > 7:
        print(f"\nLoading Step 7 results from previous run...")
        paired = pd.read_csv(os.path.join(dirs['processed'], 'paired_well_streamflow.csv'))
        paired['date'] = pd.to_datetime(paired['date'])

    # Step 8: Delta Metrics
    if start_step <= 8 <= end_step:
        data_with_deltas = run_step_8_delta_metrics(paired, dirs['features'])
    elif start_step > 8:
        print(f"\nLoading Step 8 results from previous run...")
        data_with_deltas = pd.read_csv(os.path.join(dirs['features'], 'data_with_deltas.csv'))
        data_with_deltas['date'] = pd.to_datetime(data_with_deltas['date'])

    # Step 9: Analysis
    if start_step <= 9 <= end_step:
        gage_stats, well_stats, mi_results = run_step_9_analysis(
            data_with_deltas, dirs['features'], dirs['figures']
        )


    print("\nWorkflow complete!")


if __name__ == '__main__':
    main()
