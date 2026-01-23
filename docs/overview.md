# Process Overview

GWBASE implements a 9-step workflow to analyze the relationship between groundwater levels and stream baseflow. This page describes each step in detail.

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     GWBASE WORKFLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Step 1     │    │   Step 2     │    │   Step 3     │       │
│  │   Stream     │───▶│   Locate     │───▶│   Associate  │       │
│  │   Network    │    │   Wells      │    │   Reaches    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                                       │                │
│         ▼                                       ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Step 4     │    │   Step 5     │    │   Step 6     │       │
│  │   Filter     │───▶│   PCHIP      │───▶│   Elevation  │       │
│  │   Data       │    │   Interp.    │    │   Filter     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Step 9     │    │   Step 8     │    │   Step 7     │       │
│  │   Analyze    │◀───│   Compute    │◀───│   Pair       │       │
│  │   Relations  │    │   Deltas     │    │   Records    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Identify Stream Network and Upstream Catchments

**Purpose:** Build the hydrologic framework by constructing a directed graph of the stream network and identifying terminal gages with their contributing watersheds.

**Key Operations:**
- Build a directed graph from stream segments using LINKNO (segment ID) and DSLINKNO (downstream segment ID)
- Match stream gages to their containing catchments via spatial join
- Identify terminal gages (gages with no other gages downstream)
- Delineate all upstream catchments that drain to each terminal gage

**Why Terminal Gages?** Terminal gages represent the downstream-most measurement points. Using terminal gages prevents double-counting wells that would otherwise be assigned to multiple overlapping watersheds.

**Functions:**
- `build_stream_network_graph()` - Creates NetworkX DiGraph from stream segments
- `match_gages_to_catchments()` - Spatially joins gages to catchment polygons
- `identify_terminal_gages()` - Finds gages with no downstream gages
- `delineate_all_upstream_catchments()` - Maps all upstream catchment IDs per terminal gage

**Outputs:**
- Stream network graph
- Terminal gage list
- Upstream catchment mapping (Gage_ID → list of Upstream_Catchment_IDs)

---

## Step 2: Locate Groundwater Wells within Catchments

**Purpose:** Assign each groundwater monitoring well to its terminal gage based on catchment location.

**Key Operations:**
- Perform spatial join of well points to catchment polygons
- Link wells to terminal gages using the upstream catchment mapping from Step 1
- Wells outside any mapped catchment are excluded

**Functions:**
- `locate_wells_in_catchments()` - Assigns wells to gages via catchment membership

**Outputs:**
- Well-gage assignments (well_id, gage_id, catchment_id)

---

## Step 3: Associate Wells with Nearest Stream Segments

**Purpose:** Link each well to its nearest stream reach to obtain reach elevation for vertical filtering.

**Key Operations:**
- Project well and stream geometries to UTM for accurate distance calculation
- Find the nearest stream segment (reach) for each well
- Extract reach elevation from DEM-derived data
- Identify the downstream gage for each well's reach

**Functions:**
- `associate_wells_with_reaches()` - Computes nearest reach and distance
- `extract_reach_centroids()` - Gets reach centroid coordinates

**Outputs:**
- Well-reach relationships (Well_ID, Reach_ID, Reach_Elevation, Distance_to_Reach)

---

## Step 4: Filter Wells with Insufficient Data

**Purpose:** Remove wells with too few observations or outlier values that could compromise analysis quality.

**Key Operations:**
- Detect outliers using combined Z-score and IQR methods
- Filter wells requiring minimum number of measurements
- Filter wells requiring minimum time span of observations

**Outlier Detection:**
- **Z-score method:** Flag values > 3 standard deviations from mean
- **IQR method:** Flag values outside 1.5× interquartile range
- Wells with flagged outliers have those observations removed

**Functions:**
- `GroundwaterOutlierDetector` - Class for well-specific outlier detection
- `detect_outliers()` - General outlier detection function
- `filter_wells_by_data_quality()` - Filters by measurement count and time span
- `clean_well_data_for_interpolation()` - Convenience function combining both

**Outputs:**
- Cleaned well time series ready for interpolation

---

## Step 5: Temporal Interpolation of Groundwater Levels (PCHIP)

**Purpose:** Interpolate irregular groundwater measurements to daily resolution for alignment with daily streamflow data.

**Method:** PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)

**Why PCHIP?**
- Preserves monotonicity between data points
- Avoids unrealistic oscillations common with cubic splines
- Maintains local extrema without introducing artificial peaks
- Produces hydrologically realistic water level variations

**Key Operations:**
- Convert measurement dates to ordinal numbers
- Fit PCHIP interpolator to each well's time series
- Generate daily interpolated values between first and last observation
- Wells with < 2 observations are skipped

**Functions:**
- `interpolate_daily_pchip()` - Core PCHIP interpolation
- `interpolate_with_well_info()` - Interpolates and merges well metadata
- `validate_interpolation()` - Compares interpolated vs original statistics

**Outputs:**
- Daily interpolated water table elevation (WTE) per well

---

## Step 6: Elevation-Based Filtering

**Purpose:** Retain only wells with water levels that indicate potential hydraulic connection to streams.

**Rationale:** Wells with water table elevations far below the stream channel are unlikely to contribute to or receive water from the stream. This filter focuses the analysis on wells where groundwater-surface water exchange is plausible.

**Filter Logic:**
- Calculate elevation difference: `delta_elev = reach_elevation - WTE`
- Positive delta_elev = WTE below streambed (losing/disconnected conditions)
- Negative delta_elev = WTE above streambed (gaining conditions)
- **Keep records where:** `delta_elev ≤ buffer_distance` (default: 30 meters)

This retains:
- All gaining stream conditions (WTE above streambed)
- Wells within buffer distance below streambed

**Functions:**
- `filter_by_elevation()` - Applies the elevation filter
- `analyze_elevation_sensitivity()` - Tests different buffer values
- `calculate_hydraulic_gradient()` - Computes gradient between well and stream

**Outputs:**
- Filtered dataset with elevation statistics
- Distribution of delta_elev values

---

## Step 7: Pair Groundwater and Streamflow Records

**Purpose:** Match daily interpolated groundwater levels with daily streamflow measurements under baseflow-dominated (BFD) conditions.

**Baseflow-Dominated Periods:** BFD periods are identified using a machine learning classifier that distinguishes baseflow conditions from event-driven (storm) conditions based on hydrograph characteristics. During BFD periods, streamflow is primarily sustained by groundwater discharge.

**Key Operations:**
- Join well time series with streamflow by gage_id and date
- Add BFD classification flag
- Calculate baseline values (WTE₀, Q₀) from first BFD occurrence per well

**Baseline Selection:** The first BFD=1 date for each well establishes:
- WTE₀ = Water table elevation at baseline
- Q₀ = Streamflow at baseline

All subsequent changes are measured relative to this baseline.

**Functions:**
- `pair_wells_with_streamflow()` - Merges well and streamflow data
- `filter_to_bfd_periods()` - Filters to BFD=1 records only
- `calculate_baseline_values()` - Computes WTE₀ and Q₀ per well

**Outputs:**
- Paired dataset (well_id, gage_id, date, wte, q, bfd, wte0, q0)

---

## Step 8: Compute ΔWTE and ΔQ

**Purpose:** Calculate changes in water table elevation and streamflow relative to baseline conditions.

**Delta Metrics:**
- **ΔWTE = WTE - WTE₀** (change in water table elevation, feet)
- **ΔQ = Q - Q₀** (change in streamflow, cfs)

**Interpretation:**
- Negative ΔWTE indicates declining water levels (groundwater depletion)
- Negative ΔQ indicates reduced streamflow (baseflow decline)
- Positive correlation between ΔWTE and ΔQ suggests groundwater-baseflow linkage

**Lag Analysis:** Groundwater changes may precede streamflow responses by weeks to years. GWBASE supports lagged analysis:
- 3-month lag
- 6-month lag
- 1-year lag
- Multi-year lags

**Functions:**
- `compute_delta_metrics()` - Calculates ΔWTE and ΔQ
- `create_lag_analysis()` - Creates lagged ΔWTE columns

**Outputs:**
- Dataset with delta_wte, delta_q columns
- Lagged versions for time-delay analysis

---

## Step 9: Analyze ΔWTE–ΔQ Relationships

**Purpose:** Quantify the statistical relationship between groundwater level changes and streamflow changes.

**Analysis Methods:**

### Linear Regression
- Fits ΔQ = slope × ΔWTE + intercept per gage or per well
- Reports R², p-value, slope, intercept
- Positive slope indicates expected positive correlation

### Mutual Information (MI)
- Non-parametric measure of dependency
- Captures both linear and non-linear relationships
- Higher MI indicates stronger association regardless of functional form

### Cross-Correlation Function (CCF)
- Identifies optimal time lag between ΔWTE and ΔQ
- Reveals whether groundwater changes lead or lag streamflow changes
- Useful for characterizing aquifer response times

**Functions:**
- `compute_regression_by_gage()` - Linear regression aggregated by gage
- `compute_regression_by_well()` - Linear regression per well-gage pair
- `compute_mi_analysis()` - Mutual information per well-gage pair
- `compute_ccf_by_watershed()` - Cross-correlation analysis
- `compare_lag_vs_no_lag()` - Compares lagged vs concurrent relationships

**Outputs:**
- Regression statistics (slope, R², p-value) per gage/well
- MI scores per well-gage pair
- Optimal lag times from CCF analysis
- Summary statistics and visualizations
