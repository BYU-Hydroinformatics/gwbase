# Step 5: PCHIP Interpolation

## Overview

Step 5 interpolates sparse groundwater observations to daily frequency using PCHIP.

## Purpose

Generate continuous daily time series from irregularly sampled groundwater measurements.

## Input Data

- Step 4 output: Filtered well time series

## Algorithm

1. For each well with ≥2 observations:
   - Generate daily date range from min to max date
   - Apply PCHIP interpolation
   - Skip wells with gaps > max_gap_days if specified
2. Combine interpolated time series for all wells

## Output

**File:** `outputs/interim/step05_wte_daily_pchip.parquet`

