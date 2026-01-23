# API Reference

Complete reference for all GWBASE functions organized by module.

---

## Data Loading

### load_hydrography_data

```python
gwbase.load_hydrography_data(
    filepath: str,
    layer: str = None
) -> gpd.GeoDataFrame
```

Load stream network or catchment shapefile.

**Parameters:**
- `filepath`: Path to shapefile or GeoPackage
- `layer`: Layer name for multi-layer files (optional)

**Returns:** GeoDataFrame with geometry and attributes

**Example:**
```python
streams = gwbase.load_hydrography_data('data/streams.shp')
catchments = gwbase.load_hydrography_data('data/catchments.gpkg', layer='catchments')
```

---

### load_well_locations

```python
gwbase.load_well_locations(
    filepath: str,
    well_id_col: str = 'Well_ID',
    lat_col: str = 'lat_dec',
    lon_col: str = 'long_dec',
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame
```

Load groundwater well locations from a CSV file (e.g., from USGS NWIS).

Creates a GeoDataFrame with point geometries from lat/lon coordinates.

**Parameters:**
- `filepath`: Path to CSV file with well locations
- `well_id_col`: Column name for well identifier (default: 'Well_ID')
- `lat_col`: Column name for latitude (default: 'lat_dec')
- `lon_col`: Column name for longitude (default: 'long_dec')
- `crs`: Coordinate reference system (default: "EPSG:4326")

**Returns:** GeoDataFrame with well locations as point geometries

**Example:**
```python
wells_gdf = gwbase.load_well_locations('data/groundwater/wells.csv')
print(f"Loaded {len(wells_gdf)} wells")
```

---

### load_groundwater_data

```python
gwbase.load_groundwater_data(
    well_locations_path: str,
    timeseries_path: str,
    well_id_col: str = 'Well_ID',
    lat_col: str = 'lat_dec',
    lon_col: str = 'long_dec'
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]
```

Load groundwater well locations and time series data from CSV files.

**Parameters:**
- `well_locations_path`: Path to CSV file with well locations (Well_ID, lat_dec, long_dec, GSE)
- `timeseries_path`: Path to CSV file with water level time series (well_id, date, wte)
- `well_id_col`: Column name for well identifier (default: 'Well_ID')
- `lat_col`: Column name for latitude (default: 'lat_dec')
- `lon_col`: Column name for longitude (default: 'long_dec')

**Returns:** Tuple of (wells_gdf, well_ts, well_info)
- `wells_gdf`: GeoDataFrame with well point geometries
- `well_ts`: DataFrame with water level time series
- `well_info`: DataFrame with well metadata (without geometry)

**Example:**
```python
wells_gdf, well_ts, well_info = gwbase.load_groundwater_data(
    well_locations_path='data/groundwater/wells.csv',
    timeseries_path='data/groundwater/water_levels.csv'
)
```

---

### load_streamflow_data

```python
gwbase.load_streamflow_data(
    directory: str,
    filename: str = 'daily_discharge.csv'
) -> pd.DataFrame
```

Load daily streamflow data.

**Parameters:**
- `directory`: Directory containing streamflow files
- `filename`: CSV filename

**Returns:** DataFrame with gage_id, date, q columns

---

### load_gage_info

```python
gwbase.load_gage_info(
    filepath: str
) -> pd.DataFrame
```

Load gage metadata (ID, name, location).

**Parameters:**
- `filepath`: Path to gage info CSV

**Returns:** DataFrame with id, name, latitude, longitude columns

---

### load_baseflow_classification

```python
gwbase.load_baseflow_classification(
    filepath: str
) -> pd.DataFrame
```

Load BFD classification results.

**Parameters:**
- `filepath`: Path to BFD CSV

**Returns:** DataFrame with gage_id, date, bfd columns

---

### load_reach_elevations

```python
gwbase.load_reach_elevations(
    filepath: str
) -> pd.DataFrame
```

Load stream reach elevation data.

**Parameters:**
- `filepath`: Path to reach elevations CSV

**Returns:** DataFrame with Reach_ID, Avg_GSE columns

---

## Network Analysis (Step 1)

### build_stream_network_graph

```python
gwbase.build_stream_network_graph(
    stream_gdf: gpd.GeoDataFrame
) -> nx.DiGraph
```

Build directed graph from stream network.

**Parameters:**
- `stream_gdf`: Stream GeoDataFrame with LINKNO, DSLINKNO columns

**Returns:** NetworkX DiGraph with edges pointing downstream

**Example:**
```python
G = gwbase.build_stream_network_graph(stream_gdf)
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
```

---

### match_gages_to_catchments

```python
gwbase.match_gages_to_catchments(
    gage_df: pd.DataFrame,
    catchment_gdf: gpd.GeoDataFrame,
    linkno_col: str = 'linkno'
) -> pd.DataFrame
```

Spatially join gages to catchment polygons.

**Parameters:**
- `gage_df`: DataFrame with id, latitude, longitude
- `catchment_gdf`: Catchment GeoDataFrame
- `linkno_col`: Column name for catchment ID

**Returns:** DataFrame with id, name, catchment_id, geometry

---

### identify_terminal_gages

```python
gwbase.identify_terminal_gages(
    matched_gages: pd.DataFrame,
    stream_graph: nx.DiGraph,
    manual_remove: List[int] = None,
    manual_add: List[int] = None
) -> List[int]
```

Find gages with no other gages downstream.

**Parameters:**
- `matched_gages`: Output from match_gages_to_catchments
- `stream_graph`: Network graph
- `manual_remove`: Gage IDs to exclude
- `manual_add`: Gage IDs to include

**Returns:** List of terminal gage IDs

---

### delineate_all_upstream_catchments

```python
gwbase.delineate_all_upstream_catchments(
    terminal_gages: pd.DataFrame,
    stream_graph: nx.DiGraph
) -> pd.DataFrame
```

Map upstream catchments for all terminal gages.

**Parameters:**
- `terminal_gages`: DataFrame with id, catchment_id
- `stream_graph`: Network graph

**Returns:** DataFrame with Gage_ID, Gage_Name, Terminal_Catchment_ID, Upstream_Catchment_ID

---

## Spatial Operations (Steps 2-3)

### locate_wells_in_catchments

```python
gwbase.locate_wells_in_catchments(
    wells_gdf: gpd.GeoDataFrame,
    catchment_gdf: gpd.GeoDataFrame,
    upstream_mapping: pd.DataFrame
) -> pd.DataFrame
```

Assign wells to terminal gages via catchment membership.

**Parameters:**
- `wells_gdf`: Well locations GeoDataFrame
- `catchment_gdf`: Catchment polygons
- `upstream_mapping`: Output from delineate_all_upstream_catchments

**Returns:** DataFrame with well_id, gage_id, catchment_id, coordinates

---

### associate_wells_with_reaches

```python
gwbase.associate_wells_with_reaches(
    wells_gdf: gpd.GeoDataFrame,
    stream_gdf: gpd.GeoDataFrame,
    gages_df: pd.DataFrame,
    reach_elevations: pd.DataFrame,
    utm_crs: str = "EPSG:32612"
) -> pd.DataFrame
```

Find nearest reach and elevation for each well.

**Parameters:**
- `wells_gdf`: Well locations
- `stream_gdf`: Stream network
- `gages_df`: Gage info with COMID_v2
- `reach_elevations`: Reach elevation data
- `utm_crs`: UTM projection for distance calculation

**Returns:** DataFrame with Well_ID, Reach_ID, Reach_Elevation, Distance_to_Reach

---

## Preprocessing (Step 4)

### GroundwaterOutlierDetector

```python
detector = gwbase.GroundwaterOutlierDetector(data: pd.DataFrame)
detector.detect_outliers(
    min_points: int = 5,
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 2.0
)
clean_data = detector.get_clean_data()
```

Specialized outlier detector for groundwater data.

---

### detect_outliers

```python
gwbase.detect_outliers(
    data: pd.DataFrame,
    column: str,
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5
) -> pd.DataFrame
```

Detect outliers using Z-score and IQR methods.

**Returns:** DataFrame with is_outlier_zscore, is_outlier_iqr, is_outlier_any columns

---

### filter_wells_by_data_quality

```python
gwbase.filter_wells_by_data_quality(
    data: pd.DataFrame,
    min_measurements: int = 2,
    min_time_span_days: int = 365
) -> pd.DataFrame
```

Filter wells by minimum data requirements.

---

### clean_well_data_for_interpolation

```python
gwbase.clean_well_data_for_interpolation(
    well_ts: pd.DataFrame,
    min_points: int = 5
) -> pd.DataFrame
```

Combined outlier removal and quality filtering.

---

## Interpolation (Step 5)

### interpolate_daily_pchip

```python
gwbase.interpolate_daily_pchip(
    well_ts: pd.DataFrame,
    well_id_col: str = 'well_id',
    date_col: str = 'date',
    value_col: str = 'wte'
) -> pd.DataFrame
```

PCHIP interpolation to daily resolution.

**Parameters:**
- `well_ts`: Well time series
- `well_id_col`: Well ID column name
- `date_col`: Date column name
- `value_col`: Value column to interpolate

**Returns:** DataFrame with daily interpolated values per well

**Example:**
```python
daily = gwbase.interpolate_daily_pchip(well_ts)
print(f"Generated {len(daily):,} daily records")
```

---

### interpolate_with_well_info

```python
gwbase.interpolate_with_well_info(
    well_ts: pd.DataFrame,
    well_info: pd.DataFrame,
    lat_col: str = 'lat_dec',
    lon_col: str = 'long_dec'
) -> pd.DataFrame
```

Interpolate and merge with well metadata.

---

## Elevation Filtering (Step 6)

### filter_by_elevation

```python
gwbase.filter_by_elevation(
    filtered_data: pd.DataFrame,
    well_reach_df: pd.DataFrame,
    distance_buffer_meters: float = 30.0,
    wte_feet_to_meters: float = 0.3048
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

Filter wells by elevation difference from stream.

**Parameters:**
- `filtered_data`: Well data with wte column (feet)
- `well_reach_df`: Well-reach data with reach_elev_m column
- `distance_buffer_meters`: Maximum distance below stream (meters)
- `wte_feet_to_meters`: Conversion factor

**Returns:** Tuple of (filtered_data, elevation_distribution_stats)

---

### analyze_elevation_sensitivity

```python
gwbase.analyze_elevation_sensitivity(
    filtered_data: pd.DataFrame,
    well_reach_df: pd.DataFrame,
    buffer_values: list = None
) -> pd.DataFrame
```

Test sensitivity to different buffer values.

**Default buffer_values:** [10, 20, 30, 50, 100]

---

## Pairing (Step 7)

### pair_wells_with_streamflow

```python
gwbase.pair_wells_with_streamflow(
    well_data: pd.DataFrame,
    streamflow_data: pd.DataFrame,
    bfd_classification: pd.DataFrame
) -> pd.DataFrame
```

Merge well time series with streamflow and BFD flags.

**Returns:** DataFrame with well_id, gage_id, date, wte, q, bfd

---

### calculate_baseline_values

```python
gwbase.calculate_baseline_values(
    paired_data: pd.DataFrame
) -> pd.DataFrame
```

Compute WTE₀ and Q₀ from first BFD=1 date per well.

**Returns:** DataFrame with wte0, q0 columns added

---

### filter_to_bfd_periods

```python
gwbase.filter_to_bfd_periods(
    paired_data: pd.DataFrame
) -> pd.DataFrame
```

Filter to records where BFD=1.

---

## Delta Metrics (Steps 8-9)

### compute_delta_metrics

```python
gwbase.compute_delta_metrics(
    paired_data: pd.DataFrame
) -> pd.DataFrame
```

Calculate ΔWTE = WTE - WTE₀ and ΔQ = Q - Q₀.

**Returns:** DataFrame with delta_wte, delta_q columns

---

### create_lag_analysis

```python
gwbase.create_lag_analysis(
    data: pd.DataFrame,
    lag_period: int,
    period_unit: str = 'years'
) -> pd.DataFrame
```

Create lagged ΔWTE values.

**Parameters:**
- `lag_period`: Number of time units to lag
- `period_unit`: 'years', 'months', or 'days'

**Returns:** DataFrame with lagged ΔWTE column

**Example:**
```python
lag_1yr = gwbase.create_lag_analysis(data, 1, 'years')
lag_6mo = gwbase.create_lag_analysis(data, 6, 'months')
```

---

### compute_regression_by_gage

```python
gwbase.compute_regression_by_gage(
    data: pd.DataFrame,
    min_observations: int = 10
) -> pd.DataFrame
```

Linear regression of ΔQ vs ΔWTE aggregated by gage.

**Returns:** DataFrame with gage_id, slope, intercept, r_squared, p_value

---

### compute_regression_by_well

```python
gwbase.compute_regression_by_well(
    data: pd.DataFrame,
    min_observations: int = 10
) -> pd.DataFrame
```

Linear regression per well-gage pair.

**Returns:** DataFrame with well_id, gage_id, slope, r_squared, pearson_r, spearman_r

---

### filter_by_correlation

```python
gwbase.filter_by_correlation(
    data: pd.DataFrame,
    well_stats: pd.DataFrame,
    percentile: float = 10
) -> pd.DataFrame
```

Filter data to top-percentile correlated wells.

---

## Advanced Analysis

### compute_mi_analysis

```python
gwbase.compute_mi_analysis(
    data: pd.DataFrame,
    n_bins: int = 10
) -> pd.DataFrame
```

Mutual information analysis for all well-gage pairs.

**Returns:** DataFrame with well_id, gage_id, mi, pearson_r, spearman_r

---

### compute_ccf_by_watershed

```python
gwbase.compute_ccf_by_watershed(
    data: pd.DataFrame,
    max_lag_years: int = 10
) -> Dict
```

Cross-correlation function analysis by watershed.

**Returns:** Nested dict {gage_id: {well_id: ccf_results}}

---

### compare_lag_vs_no_lag

```python
gwbase.compare_lag_vs_no_lag(
    no_lag_mi: pd.DataFrame,
    lag_mi: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]
```

Compare MI between lagged and non-lagged analyses.

**Returns:** (merged_results, by_gage_summary, overall_summary)

---

## Visualization

### plot_well_timeseries

```python
gwbase.plot_well_timeseries(
    data: pd.DataFrame,
    output_dir: str,
    max_wells_per_gage: int = None,
    figsize: Tuple[int, int] = (15, 8)
)
```

Create dual WTE/Q time series plots.

---

### plot_delta_scatter

```python
gwbase.plot_delta_scatter(
    data: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (12, 6)
) -> pd.DataFrame
```

Create ΔQ vs ΔWTE scatter plots with regression lines.

**Returns:** DataFrame with regression statistics per gage

---

### plot_mi_comparison

```python
gwbase.plot_mi_comparison(
    merged_mi: pd.DataFrame,
    output_dir: str
)
```

Create lag vs no-lag MI comparison plots.

---

### plot_regression_summary

```python
gwbase.plot_regression_summary(
    gage_stats: pd.DataFrame,
    output_dir: str
)
```

Create R² and slope distribution plots.

---

### plot_elevation_filter_sensitivity

```python
gwbase.plot_elevation_filter_sensitivity(
    sensitivity_results: pd.DataFrame,
    output_dir: str
)
```

Plot sensitivity analysis for elevation buffer values.
