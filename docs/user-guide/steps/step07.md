# Step 7: BFD Pairing

## Overview

Step 7 pairs daily groundwater (WTE) and streamflow (Q) data only on baseflow-dominated days.

## Purpose

Focus analysis on periods when streamflow is baseflow-dominated, reducing influence of storm events.

## Input Data

- Step 6 output: Connected wells with daily WTE
- Streamflow data from CSV files (pre-filtered for BFD=1)

## Algorithm

1. Load streamflow data (already filtered for BFD=1)
2. Merge WTE data with streamflow on gage_id and date
3. Only pairs where both WTE and Q exist on same date with BFD=1

## Output

**File:** `outputs/interim/step07_bfd_pairs.parquet`

