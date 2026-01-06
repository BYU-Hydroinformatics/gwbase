# Step 3: Well-Reach Association

## Overview

Step 3 associates each well with its nearest stream reach and records the reach ID and elevation.

## Purpose

Link wells to stream segments (reaches) for elevation-based filtering and hydraulic gradient calculations.

## Input Data

- Step 2 output: Wells with gage assignments
- Stream network shapefile
- Reach elevation data (optional CSV)

## Algorithm

1. Convert coordinates to UTM for accurate distance calculations
2. For each well, calculate distance to all stream reaches
3. Find nearest reach and record reach ID
4. Look up reach elevation from elevation data if available

## Output

**File:** `outputs/interim/step03_well_reach_links.parquet`

**Columns:**
- `well_id`: Well identifier
- `gage_id`: Terminal gage identifier
- `reach_id`: Nearest stream reach identifier
- `reach_elevation_m`: Reach elevation in meters
- `distance_to_reach_m`: Horizontal distance to nearest reach

