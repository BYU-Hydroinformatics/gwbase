"""GWBASE: Groundwater-Surface Water Interaction Analysis Library."""
__version__ = "0.1.0"

# Configuration
from gwbase.config.schema import Config

# Data loading functions
from gwbase.io.readers import (
    load_groundwater_data,
    load_streamflow_data,
    load_hydrography_data,
)

# Hydrologic processing functions
from gwbase.hydro.interp import pchip_daily_interpolation
from gwbase.hydro.deltas import calculate_delta_metrics
from gwbase.hydro.qc import filter_wells_by_quality
from gwbase.hydro.filters import filter_by_elevation_buffer
from gwbase.hydro.pairing import pair_wte_with_streamflow_bfd

# Geographic/spatial analysis functions
from gwbase.geo.terminal_gage import identify_terminal_gages
from gwbase.geo.well_catchment import assign_wells_to_upstream_catchments
from gwbase.geo.well_reach import find_nearest_reach_and_elevation

# Pipeline
from gwbase.pipeline.runner import PipelineRunner

# Create aliases for README compatibility
interpolate_daily_pchip = pchip_daily_interpolation

__all__ = [
    # Version
    "__version__",
    # Configuration
    "Config",
    # Data loading
    "load_groundwater_data",
    "load_streamflow_data",
    "load_hydrography_data",
    # Interpolation
    "pchip_daily_interpolation",
    "interpolate_daily_pchip",  # Alias for README compatibility
    # Delta metrics
    "calculate_delta_metrics",
    # Quality control
    "filter_wells_by_quality",
    "filter_by_elevation_buffer",
    # Pairing
    "pair_wte_with_streamflow_bfd",
    # Geographic analysis
    "identify_terminal_gages",
    "assign_wells_to_upstream_catchments",
    "find_nearest_reach_and_elevation",
    # Pipeline
    "PipelineRunner",
]
