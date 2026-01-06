# Step 6: Elevation Filtering

## Overview

Step 6 filters wells based on vertical distance relative to nearest reach elevation.

## Purpose

Keep only wells that are hydraulically connected to streams (WTE at or above reach elevation, or within buffer).

## Input Data

- Step 3 output: Well-reach links with elevations
- Step 5 output: Daily interpolated WTE

## Algorithm

1. Convert WTE to meters if in feet
2. Calculate vertical distance: reach_elevation - WTE
3. Keep wells where: WTE ≥ reach_elevation OR (reach_elevation - WTE) ≤ buffer
4. This translates to: delta_elev ≤ vertical_buffer_meters

## Output

**File:** `outputs/interim/step06_connected_wells.parquet`

