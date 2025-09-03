# Data Processing Methods

This document details the specific data processing techniques and algorithms used in the groundwater-baseflow correlation analysis.

## Overview

The data processing workflow transforms raw datasets into analysis-ready formats suitable for correlation analysis. The process addresses common challenges in hydrologic data analysis including irregular sampling, data gaps, quality issues, and spatial-temporal alignment.

## Groundwater Data Processing

### Initial Data Loading and Inspection

```python
# Load groundwater time series
wte_data = pd.read_csv('GSLB_1900-2023_TS_with_aquifers.csv')
wte_data['Date'] = pd.to_datetime(wte_data['Date'])
```

**Quality Assessment Steps**:
1. Date format validation and parsing
2. WTE value range checking (physical plausibility)
3. Duplicate record identification and removal
4. Well ID consistency verification

### Data Quality Filtering

#### Outlier Detection
```python
def identify_outliers(df, column='WTE', method='iqr'):
    """
    Identify outliers using Interquartile Range method
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
```

**Outlier Criteria**:
- Statistical: Beyond 1.5 × IQR from quartiles
- Physical: WTE below land surface or above reasonable maximum
- Temporal: Unrealistic changes between consecutive measurements

#### Well Selection Criteria
Wells are included in analysis based on:
- **Minimum observations**: ≥5 measurements for interpolation
- **Temporal span**: ≥2 years of data
- **Data quality**: <50% flagged measurements
- **Spatial coverage**: Within basin boundary

### Time Series Interpolation

#### PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)

**Algorithm Implementation**:
```python
from scipy.interpolate import PchipInterpolator

def pchip_interpolate_well(well_data, target_dates):
    """
    Apply PCHIP interpolation to well time series
    """
    # Remove duplicates and sort
    well_clean = well_data.drop_duplicates('Date').sort_values('Date')
    
    # Check minimum points
    if len(well_clean) < 5:
        return None
    
    # Create interpolator
    interpolator = PchipInterpolator(
        well_clean['Date'].astype('int64'),
        well_clean['WTE']
    )
    
    # Interpolate to target dates
    target_timestamps = target_dates.astype('int64')
    interpolated = interpolator(target_timestamps)
    
    return pd.Series(interpolated, index=target_dates)
```

**PCHIP Advantages**:
- **Monotonicity preservation**: Maintains trends in original data
- **No oscillations**: Avoids unrealistic fluctuations
- **Local control**: Changes in one region don't affect distant regions
- **Smooth derivatives**: Provides realistic groundwater gradients

**Interpolation Quality Control**:
- Maximum gap length: 730 days (2 years)
- Boundary conditions: No extrapolation beyond observation range
- Validation: Compare interpolated vs. observed values where available

## Streamflow Data Processing

### USGS Gage Data Loading

```python
def load_gage_data(gage_id):
    """
    Load and process USGS streamflow data
    """
    file_path = f'data/raw/streamflow/GSLB_ML/{gage_id}.csv'
    
    # Load with proper date parsing
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['datetime'])
    
    # Quality flag processing
    df['Quality'] = df['00060_00003_cd']  # Discharge quality code
    df['Discharge'] = pd.to_numeric(df['00060_00003'], errors='coerce')
    
    return df[['Date', 'Discharge', 'Quality']]
```

### Flow Data Quality Control

**Quality Code Interpretation**:
- **A**: Approved - highest quality
- **P**: Provisional - subject to revision
- **e**: Estimated
- **<**: Less than reported value
- **>**: Greater than reported value

**Quality Filtering**:
```python
def filter_streamflow_quality(df, min_quality='P'):
    """
    Filter streamflow data based on quality codes
    """
    high_quality = ['A', 'P']  # Approved and Provisional
    if min_quality == 'A':
        high_quality = ['A']
    
    return df[df['Quality'].isin(high_quality)]
```

## Spatial Data Processing

### Coordinate System Standardization

All spatial datasets are standardized to consistent coordinate systems:
- **Geographic operations**: WGS84 (EPSG:4326)
- **Distance calculations**: UTM Zone 12N (EPSG:32612)

```python
def standardize_crs(gdf, target_crs='EPSG:4326'):
    """
    Standardize coordinate reference system
    """
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf
```

### Well-Stream Distance Calculation

**Algorithm**: Minimum distance from each well to stream network
```python
def calculate_well_stream_distances(wells_gdf, streams_gdf):
    """
    Calculate minimum distance from wells to streams using UTM coordinates
    """
    # Convert to UTM for accurate distance calculation
    wells_utm = wells_gdf.to_crs('EPSG:32612')
    streams_utm = streams_gdf.to_crs('EPSG:32612')
    
    distances = []
    for well in wells_utm.itertuples():
        # Calculate distance to all stream segments
        stream_distances = streams_utm.geometry.distance(well.geometry)
        min_distance = stream_distances.min()
        distances.append(min_distance)
    
    return pd.Series(distances, index=wells_utm.index)
```

### Catchment Assignment

**Method**: Point-in-polygon analysis with network topology
```python
def assign_wells_to_catchments(wells_gdf, catchments_gdf, gages_df):
    """
    Assign wells to terminal gage catchments
    """
    # Spatial join
    wells_with_catchments = gpd.sjoin(
        wells_gdf, 
        catchments_gdf, 
        how='left', 
        predicate='within'
    )
    
    # Map to terminal gages
    terminal_mapping = create_terminal_gage_mapping(gages_df)
    wells_with_catchments['Terminal_Gage'] = wells_with_catchments['COMID'].map(
        terminal_mapping
    )
    
    return wells_with_catchments
```

## Delta Metrics Calculation

### Time Lag Selection

**Rationale**: 30-day lag captures:
- Monthly to seasonal hydrologic responses
- Averaging of high-frequency noise
- Reasonable response time for groundwater systems

```python
def calculate_delta_metrics(df, lag_days=30):
    """
    Calculate delta (change) metrics for time series
    """
    df_sorted = df.sort_values('Date')
    
    # Calculate lagged differences
    df_sorted['Delta_Q'] = df_sorted['Discharge'].diff(lag_days)
    df_sorted['Delta_WTE'] = df_sorted['WTE'].diff(lag_days)
    
    # Remove first lag_days observations (no delta possible)
    return df_sorted.iloc[lag_days:]
```

### Seasonal Classification

**Algorithm**: Date-based seasonal assignment
```python
def assign_seasons(df, date_col='Date'):
    """
    Assign seasons based on calendar dates
    """
    month = df[date_col].dt.month
    
    conditions = [
        month.isin([3, 4, 5]),    # Spring
        month.isin([6, 7, 8]),    # Summer
        month.isin([9, 10, 11]),  # Fall
        month.isin([12, 1, 2])    # Winter
    ]
    
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    df['Season'] = np.select(conditions, seasons)
    
    return df
```

## Statistical Analysis Methods

### Correlation Calculation

**Implementation**: Multiple correlation methods with quality control
```python
def calculate_correlations(x, y, methods=['pearson', 'spearman']):
    """
    Calculate multiple correlation coefficients with significance testing
    """
    from scipy.stats import pearsonr, spearmanr
    
    # Remove missing values
    mask = ~(pd.isna(x) | pd.isna(y))
    x_clean, y_clean = x[mask], y[mask]
    
    # Minimum sample size check
    if len(x_clean) < 20:
        return None
    
    results = {}
    
    if 'pearson' in methods:
        r_p, p_p = pearsonr(x_clean, y_clean)
        results['pearson'] = {'r': r_p, 'p': p_p, 'n': len(x_clean)}
    
    if 'spearman' in methods:
        r_s, p_s = spearmanr(x_clean, y_clean)
        results['spearman'] = {'r': r_s, 'p': p_s, 'n': len(x_clean)}
    
    return results
```

### Bootstrap Confidence Intervals

```python
def bootstrap_correlation(x, y, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for correlation
    """
    from scipy.stats import pearsonr
    import numpy as np
    
    correlations = []
    n = len(x)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        x_boot, y_boot = x.iloc[indices], y.iloc[indices]
        
        # Calculate correlation
        r, _ = pearsonr(x_boot, y_boot)
        correlations.append(r)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(correlations, 100 * alpha / 2)
    upper = np.percentile(correlations, 100 * (1 - alpha / 2))
    
    return lower, upper
```

## Data Export and Storage

### Processed Data Storage

**Hierarchical Storage Structure**:
```
data/
├── processed/
│   ├── wells_interpolated.csv      # PCHIP interpolated well data
│   ├── streamflow_quality.csv      # Quality-filtered gage data
│   ├── well_gage_pairs.csv         # Paired well-gage relationships
│   └── delta_metrics.csv           # Calculated delta metrics
└── features/
    ├── correlation_matrix.csv       # Correlation results
    ├── seasonal_correlations.csv    # Seasonal analysis
    └── spatial_metrics.csv          # Distance and spatial metrics
```

### Quality Metadata

Each processed dataset includes metadata on:
- Processing date and software versions
- Quality control parameters used
- Data filtering criteria applied
- Sample sizes and coverage statistics

## Performance Optimization

### Memory Management
- **Chunked processing** for large datasets
- **Lazy loading** of data only when needed
- **Garbage collection** after major processing steps

### Computational Efficiency
- **Vectorized operations** using NumPy/Pandas
- **Parallel processing** for independent calculations
- **Caching** of intermediate results

### Quality Assurance

**Automated Testing**:
- Unit tests for individual processing functions
- Integration tests for complete workflows
- Regression tests to ensure consistent results

**Manual Validation**:
- Visual inspection of time series plots
- Spot checks of correlation calculations
- Physical reasonableness assessment