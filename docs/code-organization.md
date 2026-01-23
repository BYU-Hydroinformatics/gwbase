# Code Organization

This page describes the structure of the GWBASE codebase and the purpose of each module.

## Directory Structure

```
gwbase/
├── main_gwbase.py          # Main workflow script
├── gwbase/                  # Python package
│   ├── __init__.py         # Package initialization and exports
│   ├── data_loading.py     # Data loading functions
│   ├── network.py          # Step 1: Stream network analysis
│   ├── spatial.py          # Steps 2-3: Spatial operations
│   ├── preprocessing.py    # Step 4: Outlier detection
│   ├── interpolation.py    # Step 5: PCHIP interpolation
│   ├── filtering.py        # Step 6: Elevation filtering
│   ├── pairing.py          # Step 7: Well-gage pairing
│   ├── metrics.py          # Steps 8-9: Delta metrics
│   ├── analysis.py         # Advanced statistical analysis
│   └── visualization.py    # Plotting functions
├── notebooks/              # Jupyter notebooks (exploratory)
├── data/                   # Data directory (see Data Guide)
├── reports/                # Output reports and figures
└── docs/                   # Documentation
```

---

## Module Descriptions

### main_gwbase.py

The main entry point for running the complete GWBASE workflow. Contains:

- `setup_directories()` - Creates required directory structure
- `run_step_1_network_analysis()` - Executes Step 1
- `run_step_2_locate_wells()` - Executes Step 2
- `run_step_3_associate_reaches()` - Executes Step 3
- `run_step_4_preprocessing()` - Executes Step 4
- `run_step_5_interpolation()` - Executes Step 5
- `run_step_6_elevation_filtering()` - Executes Step 6
- `run_step_7_pairing()` - Executes Step 7
- `run_step_8_delta_metrics()` - Executes Step 8
- `run_step_9_analysis()` - Executes Step 9
- `main()` - Command-line interface

---

### gwbase/__init__.py

Package initialization that exposes all public functions. Allows convenient imports:

```python
import gwbase

# All functions available at package level
gwbase.interpolate_daily_pchip(data)
gwbase.compute_delta_metrics(data)
```

---

### gwbase/data_loading.py

Functions for loading various data types used in GWBASE.

| Function | Description |
|----------|-------------|
| `load_hydrography_data(filepath)` | Load stream or catchment shapefiles |
| `load_groundwater_data(directory, ...)` | Load well locations, time series, and metadata |
| `load_streamflow_data(directory)` | Load daily streamflow data |
| `load_gage_info(filepath)` | Load gage metadata (location, name) |
| `load_baseflow_classification(filepath)` | Load BFD classification results |
| `load_reach_elevations(filepath)` | Load stream reach elevation data |

---

### gwbase/network.py

**Step 1:** Stream network graph construction and terminal gage identification.

| Function | Description |
|----------|-------------|
| `build_stream_network_graph(stream_gdf)` | Build NetworkX DiGraph from stream segments |
| `match_gages_to_catchments(gage_df, catchment_gdf)` | Spatial join of gages to catchments |
| `identify_terminal_gages(matched_gages, graph)` | Find gages with no downstream gages |
| `get_upstream_catchments(gage_id, catchment_id, graph)` | Get all upstream catchment IDs |
| `delineate_all_upstream_catchments(terminal_gages, graph)` | Map upstream catchments for all terminal gages |
| `find_downstream_gage(reach_id, stream_gdf, gages_df)` | Find gage downstream of a reach |

**Key Data Structures:**
- Stream GeoDataFrame must have `LINKNO` and `DSLINKNO` columns
- Graph edges point from upstream to downstream (flow direction)

---

### gwbase/spatial.py

**Steps 2-3:** Spatial operations for well-catchment and well-reach associations.

| Function | Description |
|----------|-------------|
| `extract_reach_centroids(stream_gdf)` | Get centroid coordinates for each reach |
| `locate_wells_in_catchments(wells_gdf, catchment_gdf, upstream_mapping)` | Assign wells to terminal gages |
| `associate_wells_with_reaches(wells_gdf, stream_gdf, gages_df, reach_elevations)` | Find nearest reach and elevation per well |
| `merge_well_reach_data(well_pchip, well_reach, gage_to_wells, gage_info)` | Combine well time series with spatial data |

**Coordinate Systems:**
- Distance calculations use UTM projection (default: EPSG:32612)
- Geographic coordinates (EPSG:4326) used for storage and display

---

### gwbase/preprocessing.py

**Step 4:** Data quality filtering and outlier detection.

| Class/Function | Description |
|----------------|-------------|
| `SimpleOutlierDetector` | Base class for outlier detection |
| `GroundwaterOutlierDetector` | Specialized detector for well data |
| `detect_outliers(data, column)` | Detect outliers using Z-score and IQR |
| `filter_wells_by_data_quality(data, min_measurements, min_time_span)` | Filter by data quantity |
| `clean_well_data_for_interpolation(well_ts)` | Combined cleaning function |

**Outlier Methods:**
- Z-score: Flag values > `zscore_threshold` (default: 3.0) standard deviations
- IQR: Flag values outside Q1 - 1.5×IQR to Q3 + 1.5×IQR range

---

### gwbase/interpolation.py

**Step 5:** PCHIP interpolation to daily resolution.

| Function | Description |
|----------|-------------|
| `interpolate_daily_pchip(well_ts)` | Core PCHIP interpolation per well |
| `interpolate_with_well_info(well_ts, well_info)` | Interpolate and merge well metadata |
| `validate_interpolation(original, interpolated)` | Compare statistics pre/post interpolation |

**PCHIP Details:**
- Uses `scipy.interpolate.PchipInterpolator`
- Dates converted to ordinal numbers for interpolation
- Wells with < 2 observations are skipped

---

### gwbase/filtering.py

**Step 6:** Elevation-based filtering for hydraulic connectivity.

| Function | Description |
|----------|-------------|
| `filter_by_elevation(data, well_reach_df, buffer_m)` | Apply elevation filter |
| `analyze_elevation_sensitivity(data, well_reach_df, buffer_values)` | Test sensitivity to buffer |
| `calculate_hydraulic_gradient(data)` | Compute gradient between well and stream |

**Filter Logic:**
```
delta_elev = reach_elevation_m - wte_meters
Keep if: delta_elev <= buffer_distance (default: 30m)
```

---

### gwbase/pairing.py

**Step 7:** Pairing wells with streamflow under BFD conditions.

| Function | Description |
|----------|-------------|
| `pair_wells_with_streamflow(well_data, streamflow, bfd_class)` | Merge well and streamflow data |
| `filter_to_bfd_periods(paired_data)` | Filter to BFD=1 records |
| `calculate_baseline_values(paired_data)` | Compute WTE₀ and Q₀ per well |
| `apply_date_range_filter(data, start, end)` | Filter to date range |
| `get_well_gage_summary(paired_data)` | Summary statistics per pair |

**Baseline Calculation:**
- First occurrence of BFD=1 per well defines baseline
- WTE₀ = WTE at first BFD date
- Q₀ = Q at first BFD date

---

### gwbase/metrics.py

**Steps 8-9:** Delta metrics computation and regression analysis.

| Function | Description |
|----------|-------------|
| `compute_delta_metrics(paired_data)` | Calculate ΔWTE and ΔQ |
| `create_lag_analysis(data, lag_period, period_unit)` | Create lagged ΔWTE columns |
| `compute_regression_by_gage(data)` | Linear regression aggregated by gage |
| `compute_regression_by_well(data)` | Linear regression per well-gage pair |
| `filter_by_correlation(data, well_stats, percentile)` | Filter to top-correlated wells |
| `summarize_regression_results(gage_stats, well_stats)` | Generate summary statistics |

**Lag Options:**
- `period_unit='years'`: 1, 2, 3+ year lags
- `period_unit='months'`: 3, 6 month lags
- `period_unit='days'`: Custom day lags

---

### gwbase/analysis.py

Advanced statistical analysis methods.

| Function | Description |
|----------|-------------|
| `calculate_mutual_info(x, y, n_bins)` | Compute mutual information |
| `calculate_well_metrics(well_data)` | MI + Pearson + Spearman for one well |
| `compute_mi_analysis(data)` | MI analysis for all well-gage pairs |
| `calculate_ccf(x, y, max_lag_days)` | Cross-correlation between two series |
| `compute_ccf_by_watershed(data, max_lag_years)` | CCF analysis per watershed |
| `compare_lag_vs_no_lag(no_lag_mi, lag_mi)` | Compare lagged vs concurrent MI |
| `aggregate_ccf_results(ccf_results)` | Summarize CCF into DataFrame |

**Mutual Information:**
- Uses discretization (default: 10 bins)
- Captures non-linear relationships
- Returns score in bits

---

### gwbase/visualization.py

Plotting functions for results visualization.

| Function | Description |
|----------|-------------|
| `plot_well_timeseries(data, output_dir)` | Dual WTE/Q time series plots |
| `plot_delta_scatter(data, output_dir)` | ΔQ vs ΔWTE scatter with regression |
| `plot_mi_comparison(merged_mi, output_dir)` | Lag vs no-lag MI comparison |
| `plot_regression_summary(gage_stats, output_dir)` | R² and slope distributions |
| `plot_elevation_filter_sensitivity(results, output_dir)` | Sensitivity analysis plot |

**Output Formats:**
- PNG files at 150-200 DPI
- Organized by plot type in subdirectories

---

## Import Patterns

### Import Entire Package

```python
import gwbase

# Access functions via package
graph = gwbase.build_stream_network_graph(stream_gdf)
daily = gwbase.interpolate_daily_pchip(well_ts)
```

### Import Specific Functions

```python
from gwbase import (
    build_stream_network_graph,
    interpolate_daily_pchip,
    compute_delta_metrics,
    compute_regression_by_gage,
)
```

### Import Specific Modules

```python
from gwbase import network, interpolation, metrics

graph = network.build_stream_network_graph(stream_gdf)
daily = interpolation.interpolate_daily_pchip(well_ts)
```
