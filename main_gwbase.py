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


def setup_directories(base_dir: str) -> dict:
    """Set up directory structure for GWBASE analysis."""
    dirs = {
        'raw': os.path.join(base_dir, 'data', 'raw'),
        'processed': os.path.join(base_dir, 'data', 'processed'),
        'features': os.path.join(base_dir, 'data', 'features'),
        'figures': os.path.join(base_dir, 'reports', 'figures'),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

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
    buffer_m: float = 30.0
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

    # Save results
    filtered_data.to_csv(os.path.join(output_dir, 'filtered_by_elevation.csv'), index=False)
    dist_stats.to_csv(os.path.join(output_dir, 'elevation_distribution.csv'), index=False)

    print(f"\nStep 6 complete. {len(filtered_data):,} records retained.")

    return filtered_data


def run_step_7_pairing(
    well_data: pd.DataFrame,
    streamflow_data: pd.DataFrame,
    bfd_classification: pd.DataFrame,
    output_dir: str
):
    """
    Step 7: Pair Groundwater and Streamflow Records

    - Match well and streamflow by gage and date
    - Add BFD classification
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

    args = parser.parse_args()

    print("\n" + "="*60)
    print("GWBASE - Groundwater-Baseflow Analysis System")
    print(f"Version: {gwbase.__version__}")
    print("="*60)

    # Setup directories
    dirs = setup_directories(args.data_dir)

    print(f"\nData directory: {args.data_dir}")
    print(f"Processing steps: {args.steps}")

    # Note: This is a template. Users will need to:
    # 1. Load their specific data files
    # 2. Modify paths as needed
    # 3. Run individual steps or the complete workflow

    print("\n" + "-"*60)
    print("GWBASE workflow template ready.")
    print("Load your data and call the appropriate step functions.")
    print("-"*60)

    # Example workflow (commented out - uncomment with actual data):

    # Load data
    stream_gdf = gwbase.load_hydrography_data(
        os.path.join(dirs['raw'], 'hydrography/gslb_stream.shp')
    )
    catchment_gdf = gwbase.load_hydrography_data(
        os.path.join(dirs['raw'], 'hydrography/gsl_catchment.shp')
    )
    gage_df = gwbase.load_gage_info(
        os.path.join(dirs['raw'], 'streamflow/gsl_nwm_gage.csv')
    )
    wells_gdf, well_ts, well_info = gwbase.load_groundwater_data(
        well_locations_path=os.path.join(dirs['raw'], 'groundwater/GSLB_1900-2023_wells_with_aquifers.csv'),
        timeseries_path=os.path.join(dirs['raw'], 'groundwater/GSLB_1900-2023_TS_with_aquifers.csv')
    )
    streamflow = gwbase.load_streamflow_data(
        os.path.join(dirs['raw'], 'streamflow/GSLB_ML')
    )
    bfd_class = gwbase.load_baseflow_classification(
        os.path.join(dirs['raw'], 'bfd/bfd_classification.csv')
    )
    reach_elev = gwbase.load_reach_elevations(
        os.path.join(dirs['raw'], 'streamflow/reach_centroids_with_Elev.csv')
    )

    # Run workflow
    G, matched_gages, terminal_gages, upstream_df = run_step_1_network_analysis(
        stream_gdf, catchment_gdf, gage_df, dirs['processed']
    )

    wells_with_gages = run_step_2_locate_wells(
        wells_gdf, catchment_gdf, upstream_df, dirs['processed']
    )

    well_reach = run_step_3_associate_reaches(
        wells_gdf, stream_gdf, gage_df, reach_elev, dirs['processed']
    )

    clean_data = run_step_4_preprocessing(well_ts, dirs['processed'])

    daily_data = run_step_5_interpolation(clean_data, well_info, dirs['processed'])

    filtered_data = run_step_6_elevation_filtering(
        daily_data, well_reach, dirs['processed'], args.buffer
    )

    paired = run_step_7_pairing(filtered_data, streamflow, bfd_class, dirs['processed'])

    data_with_deltas = run_step_8_delta_metrics(paired, dirs['features'])

    gage_stats, well_stats, mi_results = run_step_9_analysis(
        data_with_deltas, dirs['features'], dirs['figures']
    )


    print("\nWorkflow complete!")


if __name__ == '__main__':
    main()
