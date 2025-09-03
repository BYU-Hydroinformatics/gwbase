# Figures and Visualizations

This page provides an overview of the key figures and visualizations generated from the groundwater-baseflow correlation analysis.

## Generated Figures Location

All figures are generated in the `reports/figures/` directory with the following organization:

```
reports/figures/
├── enhanced_gage_maps/           # Individual gage watershed maps
├── enhanced_terminal_gages_watersheds.png  # Overview map
├── monthly/                      # Monthly correlation analysis
├── scatter_plots_delta_q_delta_wte/  # Delta analysis scatter plots
├── seasonal/                     # Seasonal analysis results
└── well_timeseries_by_gage/     # Well time series grouped by gage
```

## Key Visualizations

### 1. Basin Overview Maps

#### Enhanced Terminal Gages and Watersheds
**File**: `enhanced_terminal_gages_watersheds.png`

This overview map shows:
- Great Salt Lake Basin boundary
- Terminal stream gages and their upstream catchments
- Stream network and major water bodies
- GAGES-II classification (reference vs. non-reference gages)

### 2. Individual Gage Analysis

#### Enhanced Gage Maps
**Location**: `enhanced_gage_maps/`

Individual maps for each terminal gage showing:
- Watershed boundary for the specific gage
- Associated groundwater wells
- Stream network within the catchment
- Topographic context

Example gages include:
- `gage_10126000_BEAR RIVER NEAR CORINNE - UT.png`
- `gage_10141000_WEBER RIVER NEAR PLAIN CITY - UT.png`
- `gage_10163000_PROVO RIVER AT PROVO - UT.png`

### 3. Time Series Analysis

#### Well Time Series by Gage
**Location**: `well_timeseries_by_gage/`

These plots show:
- Groundwater elevation time series for wells associated with each gage
- Individual well responses vs. aggregated patterns
- Data availability and quality indicators

Key files:
- `gage_10163000_wells_timeseries.png`: Provo River gage analysis
- `gage_10141000_wells_timeseries.png`: Weber River gage analysis
- `gage_summary_statistics.csv`: Summary statistics for all gages

#### Individual Well Analysis
- `gage_10163000_individual_wells.png`: Detailed view of individual wells
- `gage_10168000_individual_wells.png`: Little Cottonwood Creek wells

### 4. Correlation Analysis

#### Delta Q vs Delta WTE Scatter Plots
**Location**: `scatter_plots_delta_q_delta_wte/`

These scatter plots analyze the relationship between changes in:
- **Delta Q**: Change in streamflow
- **Delta WTE**: Change in water table elevation

Files include:
- `gage_10141000.png`: Weber River correlation analysis
- `gage_10152000.png`: Spanish Fork correlation analysis
- `gage_10163000.png`: Provo River correlation analysis
- `gage_10168000.png`: Little Cottonwood Creek analysis
- `scatter_delta_q_delta_wte_statistics_test.csv`: Statistical results

### 5. Seasonal Analysis

#### Seasonal Delta Analysis
**Location**: `seasonal/delta_q_vs_delta_wte_by_gage_seasonal/`

Seasonal breakdown of correlations for each gage:

**Weber River (10141000)**:
- `gage_10141000_Fall_delta_q_vs_delta_wte.png`
- `gage_10141000_Spring_delta_q_vs_delta_wte.png`
- `gage_10141000_Summer_delta_q_vs_delta_wte.png`
- `gage_10141000_Winter_delta_q_vs_delta_wte.png`

**Provo River (10163000)**: Similar seasonal files
**Other gages**: Complete seasonal analysis for major terminal gages

**Summary Statistics**: `delta_q_vs_delta_wte_seasonal_statistics.csv`

### 6. Monthly Analysis

#### Monthly Correlation Patterns
**Location**: `monthly/scatter_plots_delta_q_vs_delta_wte/`

Month-by-month correlation analysis:
- `gage_10141000_monthly.png`
- `gage_10152000_monthly.png`
- `gage_10163000_monthly.png`
- `gage_10168000_monthly.png`

**Summary Statistics**: `scatter_delta_q_vs_delta_wte_monthly_statistics.csv`

### 7. Example Analysis: Provo River (Gage 10163000)

The Provo River gage provides a comprehensive example of the analysis workflow:

#### Combined Analysis
- `gage_10163000_cleaned_combined.png`: Multi-panel analysis showing:
  - Time series overlay of groundwater and streamflow
  - Correlation scatter plot
  - Statistical summaries

#### Individual Components
- `gage_10163000_cleaned_scatter.png`: Detailed scatter plot
- `gage_10163000_cleaned_timeseries.png`: Time series comparison

## Figure Interpretation Guide

### Time Series Plots
- **Blue lines**: Streamflow data
- **Red/orange lines**: Groundwater elevation data
- **Shaded areas**: Data quality indicators or confidence intervals
- **Vertical lines**: Significant events (droughts, floods)

### Scatter Plots
- **X-axis**: Typically delta streamflow (ΔQ)
- **Y-axis**: Typically delta water table elevation (ΔWTE)
- **Colors**: May represent seasons, time periods, or data quality
- **Trend lines**: Linear regression fits
- **R² values**: Correlation strength indicators

### Maps
- **Blue features**: Water bodies and streams
- **Red points**: Groundwater wells
- **Green/colored areas**: Watersheds or catchments
- **Symbols**: Different gage types or classifications

## Data Quality Indicators

Figures include various indicators of data quality:

- **Data availability bars**: Show temporal coverage
- **Quality flags**: Indicate measurement reliability
- **Sample sizes**: Number of observations for correlations
- **Confidence intervals**: Uncertainty bounds on relationships

## Using the Figures

### For Research
- Use scatter plots to identify correlation patterns
- Examine seasonal figures for process understanding
- Review time series for data quality assessment

### For Management
- Overview maps show spatial patterns for planning
- Individual gage analyses support local decision-making
- Seasonal patterns inform water management timing

### For Presentation
- High-resolution PNG files suitable for publications
- Clear legends and annotations for interpretation
- Multiple formats available for different uses

## Reproducing Figures

All figures can be reproduced using the Jupyter notebooks in the analysis workflow:

1. **Notebook 01**: Data inventory and quality plots
2. **Notebook 02**: Terminal gage identification and mapping
3. **Notebook 05**: Delta metrics and correlation analysis
4. **Notebook 06**: Comprehensive plotting and figure generation

Modify plotting parameters in the notebooks to customize figure appearance, resolution, or content focus.