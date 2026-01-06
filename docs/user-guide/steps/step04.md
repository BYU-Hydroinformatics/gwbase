# Step 4: Filter Wells (Quality Control)

## Overview

Step 4 filters wells based on data quality criteria and removes outliers.

## Purpose

Ensure wells have sufficient data for analysis and remove erroneous measurements.

## Input Data

- Raw groundwater time series CSV

## Algorithm

1. Filter by minimum observation count
2. Filter by minimum time span (days)
3. Remove outliers using Z-score method (threshold: 3.0)
4. Remove outliers using IQR method (multiplier: 1.5)
5. Remove observations flagged by either method

## Output

**File:** `outputs/interim/step04_filtered_wells.parquet`

