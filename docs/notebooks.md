# Analysis Notebooks

This section describes the Jupyter notebooks used in the groundwater-baseflow correlation analysis. The notebooks are located in the `notebooks/` directory and should be run in sequence.

## Notebook Workflow

### 1. Data Inventory (01_data_inventory.ipynb)

**Purpose**: Comprehensive assessment of data availability and quality

**Key Functions**:
- Load and inspect groundwater time series data
- Analyze measurement frequency and temporal coverage
- Identify wells with sufficient data for analysis
- Create data quality visualizations
- Generate summary statistics

**Outputs**:
- Data quality reports
- Well measurement distribution plots
- Seasonal measurement patterns
- Quality metrics summary

**Key Code Sections**:
```python
# Load groundwater data
wte_data = pd.read_csv('data/raw/groundwater/GSLB_1900-2023_TS_with_aquifers.csv')

# Quality analysis
report = analyze_wte_data_quality(wte_data)
seasonal_distribution = analyze_seasonal_distribution(wte_data)
```

### 2. Terminal Gages (02_terminal_gages.ipynb)

**Purpose**: Identify downstream gages and map catchment relationships

**Key Functions**:
- Load hydrographic data (basins, streams, gages)
- Identify terminal (downstream-most) gages in each watershed
- Create spatial maps of gage networks
- Classify gages using GAGES-II reference/non-reference categories

**Outputs**:
- Terminal gage identification
- Watershed boundary maps
- Gage classification visualizations
- Spatial relationship data

### 3. PCHIP Interpolation (03_pchip_interpolation.ipynb)

**Purpose**: Fill gaps in sparse groundwater time series using PCHIP interpolation

**Key Functions**:
- Apply Piecewise Cubic Hermite Interpolating Polynomial
- Handle irregular sampling intervals
- Quality control for interpolation results
- Generate continuous time series for correlation analysis

**Key Features**:
- Preserves monotonicity of original data
- Avoids unrealistic oscillations
- Maintains physical realism in hydrologic data

**Outputs**:
- Interpolated groundwater time series
- Quality assessment of interpolation results
- Comparison plots (original vs interpolated)

### 4. Well-Gage Pairing (04_pairing_wells_gages.ipynb)

**Purpose**: Associate groundwater wells with appropriate stream gages

**Key Functions**:
- Calculate distances from wells to stream network
- Use catchment boundaries to assign wells to gages
- Apply network topology for downstream gage identification
- Filter wells based on distance and data quality criteria

**Algorithms**:
- Minimum distance calculation (UTM coordinates for accuracy)
- Point-in-polygon analysis for catchment assignment
- Network traversal for downstream gage identification

**Outputs**:
- Well-gage pair relationships
- Distance metrics
- Spatial assignment maps
- Quality-filtered well networks

### 5. Delta Metrics (05_delta_metrics.ipynb)

**Purpose**: Calculate change-based metrics for correlation analysis

**Key Functions**:
- Compute delta (change) metrics for both groundwater and streamflow
- Apply time lag considerations (typically 30 days)
- Perform seasonal correlation analysis
- Calculate statistical significance

**Delta Metrics**:
- **ΔQ**: Change in streamflow over time lag
- **ΔWTE**: Change in water table elevation over time lag
- **Seasonal breakdown**: Spring, Summer, Fall, Winter correlations

**Statistical Analysis**:
- Pearson and Spearman correlations
- Bootstrap confidence intervals
- Significance testing
- Quality control filters

**Outputs**:
- Correlation coefficient matrices
- Seasonal correlation patterns
- Statistical significance results
- Delta metrics time series

### 6. Visualization (06_plot_per_gage.ipynb)

**Purpose**: Generate comprehensive visualizations for analysis results

**Key Functions**:
- Create individual gage analysis plots
- Generate basin-wide overview maps
- Produce correlation scatter plots
- Make time series comparisons

**Plot Types**:
- **Time Series Plots**: Groundwater vs streamflow over time
- **Scatter Plots**: Delta Q vs Delta WTE relationships
- **Maps**: Spatial distribution of correlations
- **Seasonal Analysis**: Quarterly correlation patterns
- **Statistical Summaries**: Box plots and distribution plots

**Output Organization**:
```
reports/figures/
├── enhanced_gage_maps/           # Individual gage watershed maps
├── scatter_plots_delta_q_delta_wte/  # Correlation scatter plots
├── seasonal/                     # Seasonal analysis plots
├── monthly/                      # Monthly analysis plots
└── well_timeseries_by_gage/     # Well time series by gage
```

## Running the Notebooks

### Prerequisites

Ensure you have installed all required packages:
```bash
pip install -r requirements.txt
```

### Execution Order

The notebooks should be run in numerical order (01 through 06) as each builds on results from previous notebooks.

### Data Requirements

Before running notebooks, ensure the following data files are in place:

1. **Groundwater data**: `data/raw/groundwater/`
2. **Streamflow data**: `data/raw/streamflow/GSLB_ML/`
3. **Hydrography data**: `data/raw/hydrography/`
4. **Elevation data**: `data/raw/streamflow/reach_centroids_with_Elev.csv`

### Customization

Key parameters that can be adjusted:

- **Interpolation settings**: Gap length limits, minimum observations
- **Correlation parameters**: Time lags, significance thresholds
- **Spatial filters**: Distance limits, catchment size criteria
- **Quality controls**: Data quality thresholds, temporal coverage requirements

## Output Integration

The notebooks produce intermediate data files that are used by subsequent notebooks:

```
data/processed/
├── wells_interpolated.csv        # From notebook 03
├── well_gage_pairs.csv          # From notebook 04
├── delta_metrics.csv            # From notebook 05
└── correlation_results.csv      # From notebook 05
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Large datasets may require chunked processing
2. **Missing data files**: Check file paths and data availability
3. **Coordinate system errors**: Ensure consistent CRS across spatial datasets
4. **Interpolation failures**: Check minimum data requirements for PCHIP

### Performance Tips

1. **Use subset data**: Test with smaller datasets first
2. **Monitor memory usage**: Close unused variables and dataframes
3. **Save intermediate results**: Avoid re-running expensive calculations
4. **Use parallel processing**: Where applicable for independent calculations

For detailed implementation and specific code examples, refer to the individual notebook files in the `notebooks/` directory.