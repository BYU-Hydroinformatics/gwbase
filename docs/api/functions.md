# API Reference

This page documents the key functions and classes used in the groundwater-baseflow correlation analysis.

## Data Loading Functions

### `load_groundwater_data(filepath)`
Load and parse groundwater time series data.

**Parameters:**
- `filepath` (str): Path to groundwater CSV file

**Returns:**
- `pandas.DataFrame`: Parsed groundwater data with datetime index

**Example:**
```python
wte_data = load_groundwater_data('data/raw/groundwater/GSLB_1900-2023_TS_with_aquifers.csv')
```

### `load_streamflow_data(gage_id)`
Load streamflow data for a specific USGS gage.

**Parameters:**
- `gage_id` (str): USGS gage identifier

**Returns:**
- `pandas.DataFrame`: Streamflow data with quality codes

## Data Quality Functions

### `analyze_wte_data_quality(df)`
Comprehensive quality analysis of water table elevation data.

**Parameters:**
- `df` (pandas.DataFrame): Groundwater data with columns ['Well_ID', 'Date', 'WTE']

**Returns:**
- `dict`: Quality metrics including measurement counts, time spans, and data distribution

**Quality Metrics:**
- Total number of wells
- Wells with single/few measurements
- Average measurements per well
- Temporal coverage statistics

### `identify_outliers(df, column='WTE', method='iqr')`
Identify outliers in hydrologic time series.

**Parameters:**
- `df` (pandas.DataFrame): Input data
- `column` (str): Column name to analyze
- `method` (str): Outlier detection method ('iqr', 'zscore', 'modified_zscore')

**Returns:**
- `pandas.DataFrame`: DataFrame containing only outlier records

## Interpolation Functions

### `pchip_interpolate_well(well_data, target_dates)`
Apply PCHIP interpolation to sparse groundwater time series.

**Parameters:**
- `well_data` (pandas.DataFrame): Well time series with 'Date' and 'WTE' columns
- `target_dates` (pandas.DatetimeIndex): Target dates for interpolation

**Returns:**
- `pandas.Series`: Interpolated water table elevations

**Features:**
- Preserves monotonicity of original data
- Handles irregular sampling intervals
- Quality control for minimum data requirements

## Spatial Analysis Functions

### `calculate_well_stream_distances(wells_gdf, streams_gdf)`
Calculate minimum distances from wells to stream network.

**Parameters:**
- `wells_gdf` (geopandas.GeoDataFrame): Well locations
- `streams_gdf` (geopandas.GeoDataFrame): Stream network

**Returns:**
- `pandas.Series`: Minimum distances in meters (UTM projection)

### `assign_wells_to_catchments(wells_gdf, catchments_gdf, gages_df)`
Assign wells to terminal gage catchments using spatial analysis.

**Parameters:**
- `wells_gdf` (geopandas.GeoDataFrame): Well locations
- `catchments_gdf` (geopandas.GeoDataFrame): Catchment boundaries
- `gages_df` (pandas.DataFrame): Gage metadata

**Returns:**
- `geopandas.GeoDataFrame`: Wells with assigned terminal gages

### `find_downstream_gage(reach_id, stream_gdf, gages_df)`
Find downstream gage using network topology.

**Parameters:**
- `reach_id` (int): Starting stream reach identifier
- `stream_gdf` (geopandas.GeoDataFrame): Stream network with topology
- `gages_df` (pandas.DataFrame): Gage locations and attributes

**Returns:**
- `str` or `None`: Downstream gage ID if found

**Algorithm:**
- Follows downstream links in stream network
- Terminates at first gage encountered or network end
- Prevents infinite loops with visited reach tracking

## Delta Metrics Functions

### `calculate_delta_metrics(df, lag_days=30)`
Calculate change metrics for correlation analysis.

**Parameters:**
- `df` (pandas.DataFrame): Time series data with 'Date', 'Discharge', 'WTE' columns
- `lag_days` (int): Time lag for difference calculation

**Returns:**
- `pandas.DataFrame`: Data with added 'Delta_Q' and 'Delta_WTE' columns

### `assign_seasons(df, date_col='Date')`
Assign seasonal categories to time series data.

**Parameters:**
- `df` (pandas.DataFrame): Input data with date column
- `date_col` (str): Name of date column

**Returns:**
- `pandas.DataFrame`: Data with added 'Season' column

**Seasonal Definitions:**
- Spring: March-May
- Summer: June-August
- Fall: September-November
- Winter: December-February

## Correlation Analysis Functions

### `calculate_correlations(x, y, methods=['pearson', 'spearman'])`
Calculate correlation coefficients with significance testing.

**Parameters:**
- `x` (pandas.Series): First variable (e.g., Delta_Q)
- `y` (pandas.Series): Second variable (e.g., Delta_WTE)
- `methods` (list): Correlation methods to calculate

**Returns:**
- `dict`: Correlation results with coefficients, p-values, and sample sizes

### `bootstrap_correlation(x, y, n_bootstrap=1000, confidence=0.95)`
Calculate bootstrap confidence intervals for correlations.

**Parameters:**
- `x, y` (pandas.Series): Input variables
- `n_bootstrap` (int): Number of bootstrap samples
- `confidence` (float): Confidence level (0-1)

**Returns:**
- `tuple`: (lower_bound, upper_bound) confidence interval

### `seasonal_correlation_analysis(df, groupby_col='Season')`
Perform correlation analysis by seasonal groups.

**Parameters:**
- `df` (pandas.DataFrame): Data with delta metrics and seasonal assignments
- `groupby_col` (str): Column for grouping (e.g., 'Season', 'Year')

**Returns:**
- `pandas.DataFrame`: Correlation results by group

## Visualization Functions

### `plot_well_timeseries(well_data, gage_data=None, title=None)`
Create time series plots for groundwater and streamflow data.

**Parameters:**
- `well_data` (pandas.DataFrame): Well time series
- `gage_data` (pandas.DataFrame, optional): Corresponding gage data
- `title` (str, optional): Plot title

**Returns:**
- `matplotlib.figure.Figure`: Time series plot

### `plot_correlation_scatter(x, y, labels=None, title=None)`
Create scatter plot for correlation analysis.

**Parameters:**
- `x, y` (pandas.Series): Variables to plot
- `labels` (dict, optional): Axis labels
- `title` (str, optional): Plot title

**Returns:**
- `matplotlib.figure.Figure`: Scatter plot with regression line

### `plot_basin_overview(basin_gdf, wells_gdf, gages_df, streams_gdf=None)`
Create overview map of basin with wells and gages.

**Parameters:**
- `basin_gdf` (geopandas.GeoDataFrame): Basin boundary
- `wells_gdf` (geopandas.GeoDataFrame): Well locations
- `gages_df` (pandas.DataFrame): Gage locations
- `streams_gdf` (geopandas.GeoDataFrame, optional): Stream network

**Returns:**
- `matplotlib.figure.Figure`: Basin overview map

## Utility Functions

### `create_terminal_gage_mapping(stream_gdf, gages_df)`
Create mapping from stream reaches to terminal gages.

**Parameters:**
- `stream_gdf` (geopandas.GeoDataFrame): Stream network with topology
- `gages_df` (pandas.DataFrame): Gage metadata

**Returns:**
- `dict`: Mapping from reach ID to terminal gage ID

### `filter_quality_data(df, quality_threshold='P')`
Filter hydrologic data based on quality flags.

**Parameters:**
- `df` (pandas.DataFrame): Data with quality codes
- `quality_threshold` (str): Minimum quality level

**Returns:**
- `pandas.DataFrame`: Filtered high-quality data

### `export_results(results_dict, output_dir='data/processed')`
Export analysis results to CSV files.

**Parameters:**
- `results_dict` (dict): Dictionary of analysis results
- `output_dir` (str): Output directory path

**Returns:**
- `None`: Files saved to specified directory

## Classes

### `CorrelationAnalyzer`
Main class for groundwater-streamflow correlation analysis.

#### `__init__(self, config=None)`
Initialize analyzer with configuration parameters.

#### `load_data(self, data_paths)`
Load all required datasets for analysis.

#### `process_data(self)`
Execute complete data processing workflow.

#### `calculate_correlations(self, seasonal=True)`
Perform correlation analysis with optional seasonal breakdown.

#### `generate_plots(self, output_dir='reports/figures')`
Generate all visualization outputs.

#### `export_results(self, output_dir='data/processed')`
Export processed data and results.

**Example Usage:**
```python
# Initialize analyzer
analyzer = CorrelationAnalyzer()

# Load data
data_paths = {
    'groundwater': 'data/raw/groundwater/GSLB_1900-2023_TS_with_aquifers.csv',
    'streamflow': 'data/raw/streamflow/GSLB_ML/',
    'spatial': 'data/raw/hydrography/'
}
analyzer.load_data(data_paths)

# Process and analyze
analyzer.process_data()
results = analyzer.calculate_correlations(seasonal=True)

# Generate outputs
analyzer.generate_plots()
analyzer.export_results()
```

## Error Handling

### `DataQualityError`
Raised when data quality is insufficient for analysis.

### `SpatialMismatchError`
Raised when spatial datasets have coordinate system issues.

### `InterpolationError`
Raised when interpolation fails due to insufficient data.

## Configuration

### Default Parameters
```python
DEFAULT_CONFIG = {
    'min_observations': 20,
    'max_missing_data': 0.5,
    'interpolation_max_gap': 730,  # days
    'correlation_significance': 0.05,
    'bootstrap_samples': 1000,
    'seasonal_analysis': True,
    'quality_threshold': 'P'
}
```

### Customization
All parameters can be customized through configuration files or direct parameter passing to functions.