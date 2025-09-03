# Methodology

This document describes the methodological approach used for analyzing groundwater-baseflow correlations in the Great Salt Lake Basin.

## Research Framework

### Conceptual Model

The analysis is based on the conceptual understanding that groundwater and surface water systems are interconnected, with the degree of connection depending on:

1. **Hydrogeologic properties**: Aquifer type, permeability, and connectivity
2. **Topographic setting**: Valley vs. mountain locations, distance to streams
3. **Climate patterns**: Seasonal recharge and evapotranspiration cycles
4. **Anthropogenic influences**: Pumping, irrigation, and flow regulation

### Hypothesis

We hypothesize that:
- Wells closer to streams will show stronger correlations with streamflow
- Seasonal patterns will reflect different hydrologic processes
- Correlation strength varies with hydrogeologic setting
- Delta (change) metrics better capture short-term relationships

## Analytical Workflow

### Phase 1: Data Preparation and Quality Assessment

#### 1.1 Data Inventory
- Comprehensive assessment of data availability and quality
- Identification of temporal coverage and measurement frequency
- Spatial distribution analysis of monitoring networks

#### 1.2 Data Cleaning
- Outlier detection using statistical thresholds
- Quality flag interpretation and filtering
- Temporal consistency checks

#### 1.3 Spatial Processing
- Coordinate system standardization to consistent projection
- Spatial relationship mapping between wells and stream gages
- Network topology analysis for flow routing

### Phase 2: Temporal Alignment and Gap Filling

#### 2.1 Terminal Gage Identification
**Method**: Network topology analysis
- Identify downstream-most gages in each major watershed
- Map upstream catchment areas for each terminal gage
- Associate wells with appropriate terminal gages based on location

#### 2.2 PCHIP Interpolation
**Method**: Piecewise Cubic Hermite Interpolating Polynomial
- **Purpose**: Fill gaps in sparse groundwater time series
- **Advantages**: 
  - Preserves monotonicity of original data
  - Avoids oscillations common with spline interpolation
  - Maintains physical realism in hydrologic data
- **Implementation**: Applied to wells with minimum 5 observations
- **Quality control**: Interpolation limited to gaps < 2 years

#### 2.3 Well-Gage Pairing
**Method**: Spatial proximity and network analysis
- Calculate minimum distance from each well to stream network
- Identify upstream catchment for each gage
- Associate wells with gages based on catchment boundaries
- Filter for wells within reasonable influence distance (< 10 km)

### Phase 3: Correlation Analysis

#### 3.1 Delta Metrics Calculation
**Rationale**: Change-based metrics reduce effects of:
- Long-term trends unrelated to local interactions
- Systematic biases in measurements
- Non-stationary time series characteristics

**Delta Streamflow (ΔQ)**:
```
ΔQ(t) = Q(t) - Q(t-Δt)
```

**Delta Water Table Elevation (ΔWTE)**:
```
ΔWTE(t) = WTE(t) - WTE(t-Δt)
```

Where Δt represents the time lag (typically 30 days to capture seasonal responses)

#### 3.2 Correlation Calculation
**Methods**: Multiple correlation approaches
- **Pearson correlation**: Linear relationships
- **Spearman correlation**: Monotonic relationships
- **Lag correlation**: Time-delayed relationships

**Quality filters**:
- Minimum 20 paired observations
- Maximum 50% missing data in analysis period
- Temporal overlap requirement: minimum 2 years

#### 3.3 Seasonal Analysis
**Seasonal definitions**:
- Spring: March-May (snowmelt period)
- Summer: June-August (irrigation/ET period)
- Fall: September-November (recharge period)
- Winter: December-February (low activity period)

**Analysis approach**:
- Calculate correlations for each season separately
- Compare seasonal patterns across different hydrogeologic settings
- Identify dominant hydrologic processes by season

### Phase 4: Statistical Analysis and Visualization

#### 4.1 Spatial Pattern Analysis
- Map correlation coefficients by location
- Analyze relationship with distance to streams
- Examine effects of hydrogeologic setting

#### 4.2 Temporal Pattern Analysis
- Long-term trend analysis (multi-decadal)
- Seasonal variability assessment
- Identification of drought/wet period effects

#### 4.3 Uncertainty Assessment
- Bootstrap confidence intervals for correlations
- Sensitivity analysis to parameter choices
- Data quality impact assessment

## Quality Control Measures

### Data Quality Flags
1. **High quality**: >50 observations, <20% missing data, >10-year record
2. **Medium quality**: 20-50 observations, 20-40% missing data, 5-10-year record
3. **Low quality**: <20 observations, >40% missing data, <5-year record

### Statistical Significance
- Correlation significance testing (p < 0.05)
- Correction for multiple testing when appropriate
- Effect size assessment beyond statistical significance

### Validation Approaches
- Split-sample validation for temporal stability
- Cross-validation with independent datasets where available
- Physical reasonableness checks on correlation patterns

## Limitations and Assumptions

### Data Limitations
1. **Irregular sampling**: Most wells have irregular, infrequent measurements
2. **Temporal misalignment**: Well and gage measurement dates rarely coincide exactly
3. **Spatial scale mismatch**: Point measurements vs. catchment-integrated flows

### Methodological Assumptions
1. **Linear relationships**: Correlation analysis assumes linear associations
2. **Stationarity**: Assumes consistent relationships over time
3. **Independence**: Assumes observations are independent (may be violated by serial correlation)

### Physical Assumptions
1. **Hydraulic connection**: Assumes potential for groundwater-surface water interaction
2. **Response time**: Assumes responses occur within the analysis time window
3. **Local influence**: Assumes local wells reflect local aquifer conditions

## Innovation and Contributions

### Novel Aspects
1. **Delta metric approach**: Reduces non-stationarity effects
2. **Network-based pairing**: Uses flow topology rather than simple distance
3. **Comprehensive seasonal analysis**: Separates different hydrologic processes
4. **Multi-scale analysis**: From individual well-gage pairs to basin-wide patterns

### Methodological Advances
1. **PCHIP interpolation**: Appropriate for sparse hydrologic data
2. **Quality-weighted analysis**: Incorporates data quality in interpretation
3. **Uncertainty quantification**: Provides confidence bounds on relationships
4. **Integrated spatial-temporal analysis**: Combines spatial and temporal perspectives

## Reproducibility

### Code Documentation
- All analysis code available in Jupyter notebooks
- Clear parameter documentation and sensitivity analysis
- Version control for data processing steps

### Data Provenance
- Complete documentation of data sources
- Processing step documentation
- Quality control procedure documentation

### Result Validation
- Independent verification of key findings
- Comparison with published literature where available
- Physical plausibility assessment of all results