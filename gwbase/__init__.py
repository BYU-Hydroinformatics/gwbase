"""
GWBASE - Groundwater-Baseflow Analysis System

An algorithm for assessing the impact of groundwater decline on baseflow in US streams.

The GWBASE workflow consists of 9 steps:
1. Identify Stream Network and Upstream Catchments
2. Locate Groundwater Wells within Catchments
3. Associate Wells with Nearest Stream Segments
4. Filter Wells with Insufficient Data
5. Temporal Interpolation of Groundwater Levels (PCHIP)
6. Elevation-Based Filtering
7. Pair Groundwater and Streamflow Records under Baseflow-Dominated Conditions
8. Compute ΔWTE and ΔQ
9. Analyze ΔWTE–ΔQ Relationships
"""

__version__ = "0.1.0"
__author__ = "Xueyi Li, Norman L. Jones, Gustavious P. Williams, Amin Aghababaei, Riley C. Hales"

# Well summary and data quality
from .well_summary import (
    compute_well_summary_metrics,
    compute_global_summary,
    run_well_summary,
    analyze_wte_data_quality,
    analyze_seasonal_distribution,
)

# Data loading
from .data_loading import (
    load_hydrography_data,
    load_well_locations,
    load_groundwater_data,
    load_water_level_timeseries,
    load_streamflow_data,
    load_gage_info,
    load_baseflow_classification,
    load_reach_elevations,
)

# Step 1: Network analysis
from .network import (
    build_stream_network_graph,
    match_gages_to_catchments,
    identify_terminal_gages,
    get_upstream_catchments,
    delineate_all_upstream_catchments,
    find_downstream_gage,
)

# Steps 2-3: Spatial operations
from .spatial import (
    extract_reach_centroids,
    locate_wells_in_catchments,
    associate_wells_with_reaches,
    merge_well_reach_data,
    process_wells_and_reaches,
)

# Step 4: Preprocessing
from .preprocessing import (
    SimpleOutlierDetector,
    GroundwaterOutlierDetector,
    detect_outliers,
    filter_wells_by_data_quality,
    clean_well_data_for_interpolation,
)

# Step 5: Interpolation
from .interpolation import (
    interpolate_daily_pchip,
    interpolate_daily,
    interpolate_with_well_info,
    validate_interpolation,
)

# Step 6: Elevation filtering
from .filtering import (
    filter_by_elevation,
    filter_and_analyze_wte,
    analyze_elevation_sensitivity,
    calculate_hydraulic_gradient,
)

# Step 7: Well-gage pairing
from .pairing import (
    pair_wells_with_streamflow,
    filter_to_bfd_periods,
    calculate_baseline_values,
    apply_date_range_filter,
    get_well_gage_summary,
    aggregate_streamflow_monthly_bfd,
)

# Steps 8-9: Delta metrics and regression
from .metrics import (
    compute_delta_metrics,
    create_lag_analysis,
    compute_regression_by_gage,
    compute_regression_by_well,
    filter_by_correlation,
    filter_pairs_by_r_squared,
    summarize_regression_results,
)

# Advanced analysis (MI, CCF)
from .analysis import (
    calculate_mutual_info,
    calculate_well_metrics,
    compute_mi_analysis,
    calculate_ccf,
    compute_ccf_by_watershed,
    compare_lag_vs_no_lag,
    aggregate_ccf_results,
    compute_seasonal_monthly_analysis,
    combine_regression_summary,
    compute_mk_well_wte,
    compute_mk_streamflow,
    compute_mk_gage_wte,
    calculate_ccf_by_watershed_extended,
    calculate_overall_and_watershed_ccf,
    compare_lag_datasets,
)

# Visualization
from .visualization import (
    plot_well_timeseries,
    plot_well_timeseries_with_interpolation,
    plot_delta_scatter,
    plot_high_r2_gages,
    plot_filtered_pairs_scatter,
    plot_filtered_pairs_by_gage,
    plot_pairs_by_r2_category,
    plot_mi_comparison,
    plot_mi_results,
    plot_regression_summary,
    plot_elevation_filter_sensitivity,
    plot_seasonal_monthly_analysis,
    plot_seasonal_monthly_scatter,
    # CCF visualization (notebook 05)
    plot_correlation_lag_curves,
    plot_ccf_summary,
    plot_ccf_watershed_details,
    # Watershed mapping (notebook 07)
    get_gage_watersheds_styled,
    get_gage_terminal_basin,
    get_gage_watersheds,
    load_watershed_data,
    calculate_well_gage_correlations,
    run_watershed_correlation_analysis,
    create_clean_correlation_maps_with_watersheds,
    create_watershed_distance_maps_styled,
    create_watershed_vertical_distance_maps_styled,
    create_watershed_mi_maps_no_lag,
    create_watershed_mi_delta_maps,
    create_watershed_mi_delta_maps_fixed,
    # Terminal gage visualization (notebook 02)
    create_enhanced_watershed_visualization,
    create_enhanced_gage_maps,
    create_upstream_catchment_schematic,
    # Seasonal reporting (notebook 06)
    get_season_from_month,
    generate_simple_seasonal_report,
    # Per-gage scatter plotting (notebook 06)
    plot_delta_scatter_by_gage,
    plot_monthly_delta_scatter_by_gage,
    plot_seasonal_delta_scatter_by_gage,
    plot_monthly_timeseries_by_gage,
    plot_seasonal_timeseries_by_gage,
    plot_slope_lag_analysis,
    plot_r2_lag_analysis,
)
