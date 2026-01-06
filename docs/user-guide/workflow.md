# Pipeline Workflow Overview

The GWBASE pipeline consists of 9 sequential steps that transform raw data into relationship metrics. Each step reads from standardized intermediate outputs and writes to standardized output files.

## Workflow Diagram

```
Step 1: Terminal Gages
    ↓
Step 2: Wells in Catchments
    ↓
Step 3: Well-Reach Links
    ↓
Step 4: Filter Wells (Quality)
    ↓
Step 5: PCHIP Interpolation
    ↓
Step 6: Elevation Filtering
    ↓
Step 7: BFD Pairing
    ↓
Step 8: Delta Metrics
    ↓
Step 9: Relationship Analysis
```

## Step Descriptions

### Step 1: Terminal Gage Identification

**Purpose:** Identify downstream-most (terminal) gages and map all upstream catchments.

**Input:** Hydrography data (gages, streams, catchments)

**Output:** `step01_terminal_gages.parquet`

### Step 2: Well-Catchment Association

**Purpose:** Assign wells to terminal gages based on catchment membership.

**Input:** Step 1 output, wells shapefile

**Output:** `step02_wells_in_catchments.parquet`

### Step 3: Well-Reach Linking

**Purpose:** Associate each well with nearest stream reach and record reach elevation.

**Input:** Step 2 output, stream network, reach elevations

**Output:** `step03_well_reach_links.parquet`

### Step 4: Quality Filtering

**Purpose:** Filter wells with insufficient observations and remove outliers.

**Input:** Raw groundwater time series

**Output:** `step04_filtered_wells.parquet`

### Step 5: PCHIP Interpolation

**Purpose:** Interpolate groundwater levels to daily frequency.

**Input:** Step 4 output (filtered wells)

**Output:** `step05_wte_daily_pchip.parquet`

### Step 6: Elevation Filtering

**Purpose:** Filter wells using vertical buffer threshold relative to reach elevation.

**Input:** Step 3 output (well-reach links), Step 5 output (daily WTE)

**Output:** `step06_connected_wells.parquet`

### Step 7: BFD Pairing

**Purpose:** Pair daily groundwater and streamflow only on baseflow-dominated days.

**Input:** Step 6 output, streamflow data

**Output:** `step07_bfd_pairs.parquet`

### Step 8: Delta Metrics

**Purpose:** Compute delta metrics using first BFD day as baseline.

**Input:** Step 7 output (BFD pairs)

**Output:** `step08_delta_metrics.parquet`

### Step 9: Relationship Analysis

**Purpose:** Analyze ΔWTE–ΔQ relationships and compute metrics.

**Input:** Step 8 output (delta metrics)

**Output:** `step09_relationship_metrics.parquet`

## Data Flow

Each step writes:
1. **Interim Parquet file**: Main data output with standardized filename
2. **Summary JSON file**: Metadata including counts, date ranges, and statistics

