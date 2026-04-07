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

import os
from pathlib import Path
import pandas as pd

# Import GWBASE modules
import gwbase


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load workflow configuration from a YAML file.

    Falls back to built-in defaults if the file is missing or unreadable,
    so the workflow can still run without a config file.
    """
    defaults = {
        'network': {
            'manual_remove_gages': [10171000, 10167000],
            'manual_add_gages':    [10163000, 10153100, 10152000],
        },
        'filtering': {'elevation_buffer_m': 30.0},
        'well_quality': {'min_measurements': 20, 'min_years': 3},
        'analysis': {'exclude_gages': ['10167000', '10171000']},
    }
    try:
        import yaml
        with open(config_path, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}
        # Merge top-level sections (user values override defaults)
        for section, values in defaults.items():
            if section in user_cfg:
                defaults[section].update(user_cfg[section])
        print(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        print(f"config.yaml not found – using built-in defaults.")
    except Exception as e:
        print(f"Warning: could not parse config.yaml ({e}) – using built-in defaults.")
    return defaults


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


def add_gage_id_to_filtered_data(
    filtered_data: pd.DataFrame,
    wells_with_gages: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add gage_id to filtered_data from Step 2 (polygon containment) output.

    Step 2 is the sole authoritative source for well→gage assignment.
    Step 3's Downstream_Gage is used only for reach elevation (Step 6) and
    consistency validation, never for gage assignment.
    """
    if 'gage_id' in filtered_data.columns and filtered_data['gage_id'].notna().any():
        return filtered_data

    if wells_with_gages is None or 'gage_id' not in wells_with_gages.columns:
        raise RuntimeError(
            "add_gage_id_to_filtered_data requires Step 2 output (wells_with_gages). "
            "Re-run from step 2."
        )

    print("\nAdding gage_id to filtered_data from Step 2 (polygon containment)...")
    gage_mapping = (
        wells_with_gages[['well_id', 'gage_id']]
        .drop_duplicates()
        .assign(gage_id=lambda d: d['gage_id'].astype(str).str.replace(r'\.0$', '', regex=True))
    )
    filtered_data = pd.merge(filtered_data, gage_mapping, on='well_id', how='left')
    n_matched = filtered_data['gage_id'].notna().sum()
    n_missing = filtered_data['gage_id'].isna().sum()
    print(f"  Matched: {n_matched:,} records  |  Unmatched (dropped at normalize step): {n_missing:,}")
    return filtered_data


def normalize_and_filter_gage_ids(
    data: pd.DataFrame,
    terminal_gages: pd.DataFrame = None,
    data_name: str = "data"
) -> pd.DataFrame:
    """
    Normalize gage_id format (convert to string, remove .0 suffix) and filter to terminal gages.
    """
    if 'gage_id' not in data.columns:
        return data
    
    # Normalize gage_id format
    data = data.copy()
    data['gage_id'] = data['gage_id'].astype(str).str.replace('.0', '', regex=False)
    data['gage_id'] = data['gage_id'].replace('nan', pd.NA)
    
    # Filter to only terminal gages if available
    if terminal_gages is not None and 'id' in terminal_gages.columns:
        terminal_gage_ids = set(terminal_gages['id'].astype(str).unique())
        before_count = len(data)
        data = data[data['gage_id'].isin(terminal_gage_ids)]
        after_count = len(data)
        if before_count > 0:
            print(f"Filtered {data_name} to terminal gages: {before_count:,} -> {after_count:,} records ({after_count/before_count*100:.1f}% retained)")
    
    return data


def run_step_1_network_analysis(
    stream_gdf,
    catchment_gdf,
    gage_df,
    output_dir: str,
    manual_remove: list = None,
    manual_add: list = None,
):
    """
    Step 1: Identify Stream Network and Upstream Catchments

    - Build directed graph from stream network
    - Match gages to catchments
    - Identify terminal gages
    - Delineate upstream catchments

    Parameters
    ----------
    manual_remove : list of int, optional
        Gage IDs to remove from the terminal list (loaded from config.yaml).
    manual_add : list of int, optional
        Gage IDs to add to the terminal list (loaded from config.yaml).
    """
    print("\n" + "="*60)
    print("STEP 1: Stream Network and Upstream Catchments")
    print("="*60)

    # Build stream network graph
    G = gwbase.build_stream_network_graph(stream_gdf)

    # Match gages to catchments
    matched_gages = gwbase.match_gages_to_catchments(gage_df, catchment_gdf)

    # Identify terminal gages using adjustments from config.yaml
    terminal_ids = gwbase.identify_terminal_gages(
        matched_gages, G,
        manual_remove=manual_remove or [],
        manual_add=manual_add or [],
    )

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
    min_points: int = 20,
    min_years: int = 5
):
    """
    Step 4: Filter Wells with Insufficient Data

    - Detect and remove outliers
    - Filter wells with fewer than min_points measurements
    - Filter wells whose measurements span fewer than min_years years
    """
    print("\n" + "="*60)
    print("STEP 4: Data Preprocessing")
    print("="*60)

    # Clean well data (outlier removal)
    clean_data = gwbase.clean_well_data_for_interpolation(well_ts, min_points=min_points)

    # Filter wells by minimum number of measurements
    well_counts = clean_data.groupby('well_id').size()
    wells_enough_pts = well_counts[well_counts >= min_points].index
    before = clean_data['well_id'].nunique()
    clean_data = clean_data[clean_data['well_id'].isin(wells_enough_pts)]
    after = clean_data['well_id'].nunique()
    print(f"  Min measurements filter ({min_points}): {before} -> {after} wells")

    # Filter wells by minimum year span
    clean_data['date'] = pd.to_datetime(clean_data['date'])
    well_spans = clean_data.groupby('well_id')['date'].agg(
        lambda x: (x.max() - x.min()).days / 365.25
    )
    wells_enough_years = well_spans[well_spans >= min_years].index
    before = clean_data['well_id'].nunique()
    clean_data = clean_data[clean_data['well_id'].isin(wells_enough_years)]
    after = clean_data['well_id'].nunique()
    print(f"  Min years filter ({min_years}): {before} -> {after} wells")

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

    - PCHIP interpolation to **monthly** resolution (middle of each month)
    - Merge with well location info
    """
    print("\n" + "="*60)
    print("STEP 5: PCHIP Interpolation")
    print("="*60)

    # Interpolate (now monthly, middle-of-month values)
    monthly_data = gwbase.interpolate_with_well_info(clean_data, well_info)

    # Save results
    monthly_data.to_csv(os.path.join(output_dir, 'well_pchip_monthly.csv'), index=False)

    print(f"\nStep 5 complete. Generated {len(monthly_data):,} monthly records.")

    return monthly_data


def run_step_6_elevation_filtering(
    daily_data: pd.DataFrame,
    well_reach: pd.DataFrame,
    output_dir: str,
    buffer_m: float = 30.0,
    wells_with_gages: pd.DataFrame = None,
    terminal_gages: pd.DataFrame = None,
):
    """
    Step 6: Elevation-Based Filtering

    - Filter wells based on WTE vs stream elevation
    - Retain wells with potential stream connectivity
    - gage_id is sourced exclusively from Step 2 (wells_with_gages)
    """
    print("\n" + "="*60)
    print("STEP 6: Elevation-Based Filtering")
    print("="*60)

    # Filter by elevation
    filtered_data, dist_stats = gwbase.filter_by_elevation(
        daily_data, well_reach, distance_buffer_meters=buffer_m
    )

    # Add gage_id from Step 2 only
    filtered_data = add_gage_id_to_filtered_data(
        filtered_data, wells_with_gages
    )
    filtered_data = normalize_and_filter_gage_ids(
        filtered_data, terminal_gages, "filtered_data"
    )

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

    - Aggregate streamflow to monthly intervals (average of bfd=1 flows)
    - Match well and streamflow by gage and date
    - Add BFD classification (from streamflow data if available, or from bfd_classification)
    - Calculate baseline values
    """
    print("\n" + "="*60)
    print("STEP 7: Well-Streamflow Pairing")
    print("="*60)

    # Aggregate streamflow to monthly intervals (average of bfd=1 flows)
    print("\nAggregating streamflow to monthly intervals (bfd=1 average)...")
    streamflow_monthly = gwbase.aggregate_streamflow_monthly_bfd(streamflow_data)
    
    # Save aggregated streamflow
    if output_dir:
        streamflow_monthly.to_csv(
            os.path.join(output_dir, 'streamflow_monthly_bfd.csv'),
            index=False
        )

    # Pair data with monthly aggregated streamflow
    paired = gwbase.pair_wells_with_streamflow(
        well_data, streamflow_monthly, bfd_classification
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
    paired: pd.DataFrame,
    clean_data: pd.DataFrame,
    monthly_data: pd.DataFrame,
    output_dir: str,
    figures_dir: str,
    processed_dir: str = None,
    gage_name_map: dict = None
):
    """
    Step 9: Analyze ΔWTE–ΔQ Relationships

    - Linear regression by gage
    - Linear regression by well
    - Mutual information analysis
    - Lag vs no-lag MI comparison
    - Well time series plots
    - Seasonal and monthly analysis
    """
    print("\n" + "="*60)
    print("STEP 9: Regression and Correlation Analysis")
    print("="*60)

    # Regression by gage
    gage_stats = gwbase.compute_regression_by_gage(data_with_deltas)
    _add_gage_name(gage_stats, gage_name_map).to_csv(os.path.join(output_dir, 'regression_by_gage.csv'), index=False)

    # Regression by well
    well_stats = gwbase.compute_regression_by_well(data_with_deltas)
    _add_gage_name(well_stats, gage_name_map).to_csv(os.path.join(output_dir, 'regression_by_well.csv'), index=False)

    # Summary
    summary = gwbase.summarize_regression_results(gage_stats, well_stats)

    # Mutual information analysis
    mi_results = gwbase.compute_mi_analysis(data_with_deltas)
    _add_gage_name(mi_results, gage_name_map).to_csv(os.path.join(output_dir, 'mi_analysis.csv'), index=False)

    # Lag vs no-lag MI comparison (using 1-year lag)
    lag_1yr_path = os.path.join(output_dir, 'data_lag_1yr.csv')
    if os.path.exists(lag_1yr_path):
        lag_1yr = pd.read_csv(lag_1yr_path)
        lag_1yr['date'] = pd.to_datetime(lag_1yr['date'])
        lag_col = 'delta_wte_lag_1_year'
        if lag_col in lag_1yr.columns:
            mi_lag = gwbase.compute_mi_analysis(lag_1yr, delta_wte_col=lag_col)
            merged_mi, by_gage, lag_summary = gwbase.compare_lag_vs_no_lag(mi_results, mi_lag)
            _add_gage_name(merged_mi, gage_name_map).to_csv(os.path.join(output_dir, 'mi_lag_comparison.csv'), index=False)
            gwbase.plot_mi_comparison(merged_mi, os.path.join(figures_dir, 'mi_compare'))
        else:
            print("  Skipping lag MI comparison: lag column not found")
    else:
        print("  Skipping lag MI comparison: data_lag_1yr.csv not found")

    # MI distribution charts
    gwbase.plot_mi_results(mi_results, os.path.join(figures_dir, 'mi'))

    # High R² gages: filter and plot separately
    high_r2_filtered = gwbase.plot_high_r2_gages(
        data_with_deltas, gage_stats,
        os.path.join(figures_dir, 'scatter_plots_high_r2'),
        r2_threshold=0.1
    )
    if len(high_r2_filtered) > 0:
        high_r2_filtered.to_csv(os.path.join(output_dir, 'data_high_r2_gages.csv'), index=False)

    # Well time series with raw obs, daily interp, monthly interp
    if clean_data is not None and len(clean_data) > 0 and monthly_data is not None and len(monthly_data) > 0:
        gwbase.plot_well_timeseries_with_interpolation(
            paired,
            clean_data,
            monthly_data,
            os.path.join(figures_dir, 'well_timeseries'),
            max_wells_per_gage=15
        )
    else:
        # Fallback to simple timeseries if interpolation data not available
        gwbase.plot_well_timeseries(
            paired,
            os.path.join(figures_dir, 'well_timeseries'),
            max_wells_per_gage=15
        )

    # Seasonal and monthly analysis
    seasonal_stats, monthly_stats = gwbase.compute_seasonal_monthly_analysis(data_with_deltas)
    if len(seasonal_stats) > 0:
        _add_gage_name(seasonal_stats, gage_name_map).to_csv(os.path.join(output_dir, 'seasonal_analysis.csv'), index=False)
    if len(monthly_stats) > 0:
        _add_gage_name(monthly_stats, gage_name_map).to_csv(os.path.join(output_dir, 'monthly_analysis.csv'), index=False)
    gwbase.plot_seasonal_monthly_analysis(
        seasonal_stats, monthly_stats,
        os.path.join(figures_dir, 'seasonal_monthly')
    )
    gwbase.plot_seasonal_monthly_scatter(
        data_with_deltas,
        os.path.join(figures_dir, 'seasonal_monthly')
    )

    # Combined regression summary table (overall + seasonal + monthly)
    combined_regression = gwbase.combine_regression_summary(
        gage_stats, seasonal_stats, monthly_stats
    )
    _add_gage_name(combined_regression, gage_name_map).to_csv(
        os.path.join(output_dir, 'regression_summary_combined.csv'),
        index=False
    )
    print(f"  Combined regression table: {len(combined_regression)} rows -> regression_summary_combined.csv")

    # Scatter and regression plots
    gwbase.plot_regression_summary(gage_stats, os.path.join(figures_dir, 'regression'))
    gwbase.plot_delta_scatter(data_with_deltas, os.path.join(figures_dir, 'scatter_plots'),
                              gage_name_map=gage_name_map)

    # ── Mann-Kendall + Sen's Slope ──────────────────────────────
    print("\n--- Mann-Kendall Trend Analysis ---")

    # 1. Per-well WTE trend (using paired well data, which has monthly wte)
    mk_well = gwbase.compute_mk_well_wte(
        data_with_deltas.rename(columns={'wte': 'WTE'}),
        wte_col='WTE', min_obs=10
    )
    if len(mk_well) > 0:
        _add_gage_name(
            pd.merge(mk_well, data_with_deltas[['well_id','gage_id']].drop_duplicates(),
                     on='well_id', how='left'),
            gage_name_map
        ).to_csv(os.path.join(output_dir, 'mk_well_wte.csv'), index=False)

    # 2. Per-gage aggregated WTE trend
    mk_gage_wte = gwbase.compute_mk_gage_wte(data_with_deltas)
    if len(mk_gage_wte) > 0:
        _add_gage_name(mk_gage_wte, gage_name_map).to_csv(
            os.path.join(output_dir, 'mk_gage_wte.csv'), index=False
        )

    # 3. Per-gage streamflow trend
    sf_path = os.path.join(processed_dir, 'streamflow_monthly_bfd.csv') if processed_dir else None
    if sf_path and os.path.exists(sf_path):
        sf_monthly = pd.read_csv(sf_path)
        mk_sf = gwbase.compute_mk_streamflow(sf_monthly)
        if len(mk_sf) > 0:
            _add_gage_name(mk_sf, gage_name_map).to_csv(
                os.path.join(output_dir, 'mk_streamflow.csv'), index=False
            )

    print(f"\nStep 9 complete.")

    return gage_stats, well_stats, mi_results


def _add_gage_name(df: pd.DataFrame, gage_name_map: dict, gage_id_col: str = 'gage_id') -> pd.DataFrame:
    """Insert gage_name column immediately after gage_id, using the provided name map."""
    if not gage_name_map or gage_id_col not in df.columns:
        return df
    df = df.copy()
    names = df[gage_id_col].astype(str).map(gage_name_map)
    insert_pos = df.columns.get_loc(gage_id_col) + 1
    df.insert(insert_pos, 'gage_name', names)
    return df


def main():
    """Main entry point for GWBASE workflow."""

    # ── Configuration ──────────────────────────────────────────
    cfg = load_config('config.yaml')

    data_dir   = '.'
    output_dir = None
    steps      = 'all'

    # Values from config (can still be overridden by env vars below)
    buffer_m      = cfg['filtering']['elevation_buffer_m']
    min_wte       = cfg['well_quality']['min_measurements']
    min_years     = cfg['well_quality']['min_years']
    exclude_gages = [str(g) for g in cfg['analysis']['exclude_gages']]
    manual_remove = [int(g) for g in cfg['network']['manual_remove_gages']]
    manual_add    = [int(g) for g in cfg['network']['manual_add_gages']]

    # Environment variable overrides (take precedence over config.yaml)
    env_output_dir = os.environ.get("GWBASE_OUTPUT_DIR")
    if env_output_dir:
        output_dir = env_output_dir

    env_steps = os.environ.get("GWBASE_STEPS")
    if env_steps:
        steps = env_steps

    env_min_wte = os.environ.get("GWBASE_MIN_WTE")
    if env_min_wte is not None:
        try:
            min_wte = int(env_min_wte)
        except ValueError:
            print(f"Warning: GWBASE_MIN_WTE='{env_min_wte}' is not a valid integer. Using default {min_wte}.")

    env_min_years = os.environ.get("GWBASE_MIN_YEARS")
    if env_min_years is not None:
        try:
            min_years = int(env_min_years)
        except ValueError:
            print(f"Warning: GWBASE_MIN_YEARS='{env_min_years}' is not a valid integer. Using default {min_years}.")

    env_buffer_m = os.environ.get("GWBASE_BUFFER_M")
    if env_buffer_m is not None:
        try:
            buffer_m = float(env_buffer_m)
        except ValueError:
            print(f"Warning: GWBASE_BUFFER_M='{env_buffer_m}' is not a valid float. Using default {buffer_m}.")

    env_exclude_gages = os.environ.get("GWBASE_EXCLUDE_GAGES")
    if env_exclude_gages is not None:
        exclude_gages = [g.strip() for g in env_exclude_gages.split(',') if g.strip()]

    print("\n" + "="*60)
    print("GWBASE - Groundwater-Baseflow Analysis System")
    print(f"Version: {gwbase.__version__}")
    print("="*60)

    # Setup directories
    dirs = setup_directories(data_dir, output_dir)

    print(f"\nData directory: {data_dir}")
    print(f"Processing steps: {steps}")

    # Parse steps to run
    start_step = 1
    end_step = 9
    if steps != 'all':
        if '-' in steps:
            # Range format: "1-5" or "6-9"
            parts = steps.split('-')
            start_step = int(parts[0])
            end_step = int(parts[1]) if len(parts) > 1 else 9
        elif ',' in steps:
            # Comma-separated: "4,5,6"
            steps_to_run = [int(s.strip()) for s in steps.split(',')]
            start_step = min(steps_to_run)
            end_step = max(steps_to_run)
        else:
            # Single step: "6"
            start_step = int(steps)
            end_step = int(steps)

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

    # Build gage name lookup map {str(id): name}
    gage_name_map = dict(zip(gage_df['id'].astype(str), gage_df['name'])) if 'name' in gage_df.columns else {}

    wells_gdf, well_ts, well_info = gwbase.load_groundwater_data(
        well_locations_path=os.path.join(dirs['raw'], 'groundwater/GSLB_1900-2023_wells_with_aquifers.csv'),
        timeseries_path=os.path.join(dirs['raw'], 'groundwater/GSLB_1900-2025_TS_with_aquifers.csv')
    )
    # Load streamflow data with all records (including bfd=0) for monthly aggregation
    streamflow = gwbase.load_streamflow_data(
        os.path.join(dirs['raw'], 'streamflow/gages_with_bfd_predictions'),
        filter_bfd=False  # Keep all records for monthly bfd=1 aggregation
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
            stream_gdf, catchment_gdf, gage_df, dirs['processed'],
            manual_remove=manual_remove,
            manual_add=manual_add,
        )
    elif start_step > 1:
        # Load previous results
        print(f"\nLoading Step 1 results from previous run...")
        upstream_df = pd.read_csv(os.path.join(dirs['processed'], 'terminal_gage_upstream_catchments.csv'))
        terminal_gages = pd.read_csv(os.path.join(dirs['processed'], 'terminal_gages.csv'))
        # Rebuild graph if needed for later steps
        G = gwbase.build_stream_network_graph(stream_gdf)
        matched_gages = gwbase.match_gages_to_catchments(gage_df, catchment_gdf)

    # Apply gage exclusion filter
    if exclude_gages and terminal_gages is not None:
        before = len(terminal_gages)
        terminal_gages = terminal_gages[~terminal_gages['id'].astype(str).isin(exclude_gages)]
        after = len(terminal_gages)
        if before > after:
            print(f"\nExcluded {before - after} gage(s) from analysis: {exclude_gages}")
        if upstream_df is not None and 'Gage_ID' in upstream_df.columns:
            upstream_df = upstream_df[~upstream_df['Gage_ID'].astype(str).isin(exclude_gages)]

    # Step 2: Locate Wells
    if start_step <= 2 <= end_step:
        wells_with_gages = run_step_2_locate_wells(
            wells_gdf, catchment_gdf, upstream_df, dirs['processed']
        )
    elif start_step > 2:
        print(f"\nLoading Step 2 results from previous run...")
        wells_with_gages = pd.read_csv(os.path.join(dirs['processed'], 'wells_in_catchments.csv'))

    # Step 3: Associate Reaches
    # Only process wells that passed Step 2 (polygon containment).
    # Step 3's role is to find the nearest reach for elevation comparison in Step 6;
    # gage assignment is already authoritative from Step 2.
    if start_step <= 3 <= end_step:
        if wells_with_gages is None:
            raise RuntimeError("Step 3 requires Step 2 output. Re-run from step 2.")
        step2_well_ids = set(wells_with_gages['well_id'].astype(str).unique())
        wells_gdf_step2 = wells_gdf[
            wells_gdf['well_id'].astype(str).isin(step2_well_ids)
        ].copy()
        print(f"\nStep 3 input: {len(wells_gdf_step2)} wells (filtered to Step 2 watershed wells)")
        well_reach = run_step_3_associate_reaches(
            wells_gdf_step2, stream_gdf, gage_df, reach_elev, dirs['processed']
        )
    elif start_step > 3:
        print(f"\nLoading Step 3 results from previous run...")
        well_reach = pd.read_csv(os.path.join(dirs['processed'], 'well_reach_relationships.csv'))

    # ── Consistency check: Step 3 Downstream_Gage vs Step 2 gage_id ──────────
    # Since Step 3 now only processes Step 2 wells, Downstream_Gage should
    # agree with the gage_id from Step 2 for every well.  Mismatches indicate
    # that the stream network topology and the catchment polygon assignment
    # disagree and warrant manual inspection.
    if well_reach is not None and wells_with_gages is not None:
        wr_col = 'Well_ID' if 'Well_ID' in well_reach.columns else 'well_id'
        dg_col = 'Downstream_Gage' if 'Downstream_Gage' in well_reach.columns else None

        if dg_col is not None:
            wr_check = well_reach[[wr_col, dg_col]].copy()
            wr_check.columns = ['well_id', 'downstream_gage']
            wr_check['well_id'] = wr_check['well_id'].astype(str).str.replace(r'\.0$', '', regex=True)
            wr_check['downstream_gage'] = wr_check['downstream_gage'].astype(str).str.replace(r'\.0$', '', regex=True)

            s2_check = wells_with_gages[['well_id', 'gage_id']].copy()
            s2_check['well_id'] = s2_check['well_id'].astype(str).str.replace(r'\.0$', '', regex=True)
            s2_check['gage_id']  = s2_check['gage_id'].astype(str).str.replace(r'\.0$', '', regex=True)

            merged_check = wr_check.merge(s2_check, on='well_id', how='inner')
            mismatches = merged_check[
                merged_check['downstream_gage'] != merged_check['gage_id']
            ]
            if mismatches.empty:
                print("\nConsistency check PASSED: Step 3 Downstream_Gage agrees with Step 2 gage_id for all wells.")
            else:
                print(f"\nConsistency check WARNING: {len(mismatches)} well(s) have mismatched gage assignments.")
                print("  Step 2 (polygon containment) is authoritative; Step 3 Downstream_Gage is advisory.")
                print(mismatches[['well_id', 'gage_id', 'downstream_gage']].to_string(index=False))
                mismatches.to_csv(
                    os.path.join(dirs['processed'], 'step3_gage_mismatch.csv'), index=False
                )
                print(f"  Mismatches saved to: step3_gage_mismatch.csv")

    # Step 4: Preprocessing
    if start_step <= 4 <= end_step:
        clean_data = run_step_4_preprocessing(well_ts, dirs['processed'], min_points=min_wte, min_years=min_years)
    elif start_step > 4:
        print(f"\nLoading Step 4 results from previous run...")
        clean_data = pd.read_csv(os.path.join(dirs['processed'], 'well_ts_cleaned.csv'))

    # Step 5: Interpolation
    if start_step <= 5 <= end_step:
        daily_data = run_step_5_interpolation(clean_data, well_info, dirs['processed'])
    elif start_step > 5:
        print(f"\nLoading Step 5 results from previous run...")
        daily_data = pd.read_csv(os.path.join(dirs['processed'], 'well_pchip_monthly.csv'))
        daily_data['date'] = pd.to_datetime(daily_data['date'])

    # Step 6: Elevation Filtering
    if start_step <= 6 <= end_step:
        filtered_data = run_step_6_elevation_filtering(
            daily_data, well_reach, dirs['processed'], buffer_m, wells_with_gages, terminal_gages
        )
    else:
        print(f"\nLoading Step 6 results from previous run...")
        filtered_data = pd.read_csv(os.path.join(dirs['processed'], 'filtered_by_elevation.csv'))
        filtered_data['date'] = pd.to_datetime(filtered_data['date'])
        
        # Add gage_id if missing (Step 2 is the sole source)
        filtered_data = add_gage_id_to_filtered_data(
            filtered_data, wells_with_gages
        )
        filtered_data = normalize_and_filter_gage_ids(
            filtered_data, terminal_gages, "filtered_data"
        )
        
        # Re-save if gage_id was added
        if 'gage_id' in filtered_data.columns:
            filtered_data.to_csv(os.path.join(dirs['processed'], 'filtered_by_elevation.csv'), index=False)
    
    # Normalize gage_id format and filter streamflow to terminal gages
    if streamflow is not None and 'gage_id' in streamflow.columns:
        streamflow = normalize_and_filter_gage_ids(streamflow, terminal_gages, "streamflow")

    # Step 7: Pairing
    if start_step <= 7 <= end_step:
        paired = run_step_7_pairing(filtered_data, streamflow, None, dirs['processed'])
        # Re-save with gage_name added
        _add_gage_name(paired, gage_name_map).to_csv(
            os.path.join(dirs['processed'], 'paired_well_streamflow.csv'), index=False
        )
    elif start_step > 7:
        print(f"\nLoading Step 7 results from previous run...")
        paired = pd.read_csv(os.path.join(dirs['processed'], 'paired_well_streamflow.csv'))
        paired['date'] = pd.to_datetime(paired['date'])

    # Step 8: Delta Metrics
    if start_step <= 8 <= end_step:
        data_with_deltas = run_step_8_delta_metrics(paired, dirs['features'])
        # Re-save with gage_name added
        _add_gage_name(data_with_deltas, gage_name_map).to_csv(
            os.path.join(dirs['features'], 'data_with_deltas.csv'), index=False
        )
    elif start_step > 8:
        print(f"\nLoading Step 8 results from previous run...")
        data_with_deltas = pd.read_csv(os.path.join(dirs['features'], 'data_with_deltas.csv'))
        data_with_deltas['date'] = pd.to_datetime(data_with_deltas['date'])

    # Step 9: Analysis (needs clean_data and monthly interpolated for well timeseries)
    if start_step <= 9 <= end_step:
        # Always reload data_with_deltas from CSV to avoid gage_id type issues from in-memory pipeline
        data_with_deltas = pd.read_csv(os.path.join(dirs['features'], 'data_with_deltas.csv'))
        data_with_deltas['date'] = pd.to_datetime(data_with_deltas['date'])
        # Remove outlier for gage 10163000 (delta_wte > 1400 ft)
        mask = (data_with_deltas['gage_id'].astype(str) == '10163000') & (data_with_deltas['delta_wte'].abs() > 1400)
        data_with_deltas = data_with_deltas[~mask].copy()
        # Ensure clean_data and monthly data available for well timeseries
        if clean_data is None:
            clean_data = pd.read_csv(os.path.join(dirs['processed'], 'well_ts_cleaned.csv'))
            clean_data['date'] = pd.to_datetime(clean_data['date'])
        if daily_data is None:
            daily_data = pd.read_csv(os.path.join(dirs['processed'], 'well_pchip_monthly.csv'))
            daily_data['date'] = pd.to_datetime(daily_data['date'])
        gage_stats, well_stats, mi_results = run_step_9_analysis(
            data_with_deltas, paired, clean_data, daily_data,
            dirs['features'], dirs['figures'], dirs['processed'],
            gage_name_map=gage_name_map
        )


    print("\nWorkflow complete!")


if __name__ == '__main__':
    main()
