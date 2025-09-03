# Data Overview

This document provides a comprehensive overview of the datasets used in the groundwater-baseflow correlation analysis.

## Dataset Summary

The analysis integrates multiple datasets spanning over a century of observations in the Great Salt Lake Basin:

| Dataset | Time Period | Spatial Coverage | Records |
|---------|-------------|------------------|---------|
| Groundwater Wells | 1900-2023 | Great Salt Lake Basin | ~6,000 wells |
| Streamflow Gages | Variable | Stream network | ~70 gages |
| Stream Network | Current | Basin-wide | ~6,700 reaches |
| Elevation Data | Current | Reach centroids | Multiple sources |

## Groundwater Data

### Water Table Elevation Time Series
- **File**: `GSLB_1900-2023_TS_with_aquifers.csv`
- **Description**: Historical water table elevation measurements
- **Key Columns**:
  - `Well_ID`: Unique well identifier
  - `Date`: Measurement date
  - `WTE`: Water table elevation (meters)
  - `Aquifer`: Aquifer classification

### Well Metadata
- **File**: `GSLB_1900-2023_wells_with_aquifers.csv`
- **Description**: Well locations and characteristics
- **Key Columns**:
  - `Well_ID`: Unique identifier
  - `Latitude`, `Longitude`: Well coordinates
  - `Aquifer`: Aquifer type
  - `Depth`: Well depth information

### Data Quality Characteristics

!!! info "Data Quality Summary"
    - **Wells with single measurement**: ~40% of total wells
    - **Wells with two measurements**: ~15% of total wells
    - **Average measurements per well**: Variable (1-500+ observations)
    - **Temporal coverage**: Highly variable, from single measurements to multi-decade records

## Streamflow Data

### Stream Gage Network
- **Files**: Individual CSV files in `GSLB_ML/` directory
- **Description**: Daily streamflow measurements from USGS gages
- **Naming Convention**: `{GAGE_ID}.csv`
- **Key Variables**:
  - Date
  - Streamflow (cubic feet per second)
  - Quality flags

### Gage Metadata
- **File**: `gsl_nwm_gage.csv`
- **Description**: Gage locations and characteristics
- **Key Columns**:
  - `samplingFeatureCode`: USGS gage ID
  - `latitude`, `longitude`: Gage coordinates
  - `elevation_m`: Gage elevation
  - `COMID_v2`: Associated stream reach ID

### Gage Classification
- **File**: `GAGES-II_ref_non_ref.xlsx`
- **Description**: USGS GAGES-II classification of reference vs. non-reference gages
- **Classifications**:
  - **Ref**: Reference condition (minimal human impact)
  - **Non-ref**: Non-reference condition (moderate to high human impact)

## Hydrographic Data

### Basin Boundary
- **Files**: `gsl_basin.shp` (+ .shx, .dbf, .prj)
- **Description**: Great Salt Lake Basin watershed boundary
- **Geometry**: Polygon
- **Coordinate System**: Geographic (WGS84)

### Stream Network
- **Files**: `gslb_stream.shp` (+ .shx, .dbf, .prj)
- **Description**: Stream network with flow topology
- **Key Attributes**:
  - `LINKNO`: Unique reach identifier
  - `DSLINKNO`: Downstream reach identifier
  - `USLINKNO1`: Primary upstream reach
  - `Length`: Reach length

### Catchments
- **Files**: `gsl_catchment.shp` (+ .shx, .dbf, .prj)
- **Description**: Stream reach catchment boundaries
- **Geometry**: Polygons for each stream reach
- **Attributes**: Hydrologic unit codes and areas

### Water Bodies
- **Files**: `lake.shp` (+ .shx, .dbf, .prj)
- **Description**: Lakes and reservoirs
- **Geometry**: Polygons
- **Includes**: Great Salt Lake and major reservoirs

### Well Locations (Spatial)
- **Files**: `well_shp.shp` (+ .shx, .dbf, .prj)
- **Description**: Spatial representation of groundwater wells
- **Geometry**: Points
- **Attributes**: Well IDs and basic metadata

## Elevation Data

### Stream Reach Elevations
- **File**: `reach_centroids_with_Elev.csv`
- **Description**: Elevation data for stream reach centroids
- **Key Columns**:
  - `Reach_ID`: Stream reach identifier
  - `Latitude`, `Longitude`: Centroid coordinates
  - `NASA_GSE`: NASA elevation estimate
  - `AW3D_GSE`: AW3D elevation estimate
  - `Avg_GSE`: Average elevation

!!! note "Elevation Data Sources"
    - **NASA**: NASA digital elevation model
    - **AW3D**: ALOS World 3D elevation data
    - **Avg_GSE**: Average of available elevation sources

## Data Processing Notes

### Quality Considerations

1. **Groundwater Data**:
   - Highly irregular sampling frequencies
   - Many wells with limited observations
   - Potential measurement errors and outliers
   - Missing data requires interpolation

2. **Streamflow Data**:
   - Generally high-quality USGS data
   - Some gaps in historical records
   - Flow regulation affects natural baseflow signals

3. **Spatial Data**:
   - Coordinate system consistency verified
   - Topology validation performed for stream network
   - Elevation data compared across sources

### Preprocessing Steps

The analysis workflow includes several preprocessing steps:

1. **Data cleaning**: Removal of obvious outliers and erroneous values
2. **Temporal alignment**: Synchronization of observation periods
3. **Spatial joining**: Association of wells with stream reaches and gages
4. **Gap filling**: PCHIP interpolation for sparse groundwater records
5. **Quality flagging**: Identification of low-quality data periods

## Usage in Analysis

Each dataset plays a specific role in the correlation analysis:

- **Groundwater data**: Primary response variable (WTE changes)
- **Streamflow data**: Streamflow and baseflow calculation
- **Stream network**: Spatial relationships and flow routing
- **Elevation data**: Hydraulic gradient calculations
- **Catchments**: Spatial aggregation and watershed delineation

For detailed information about how these datasets are used in the analysis workflow, see the individual notebook documentation.