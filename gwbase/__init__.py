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
    interpolate_with_well_info,
    validate_interpolation,
)

# Step 6: Elevation filtering
from .filtering import (
    filter_by_elevation,
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
)

# Steps 8-9: Delta metrics and regression
from .metrics import (
    compute_delta_metrics,
    create_lag_analysis,
    compute_regression_by_gage,
    compute_regression_by_well,
    filter_by_correlation,
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
)

# Visualization
from .visualization import (
    plot_well_timeseries,
    plot_delta_scatter,
    plot_mi_comparison,
    plot_regression_summary,
    plot_elevation_filter_sensitivity,
)
