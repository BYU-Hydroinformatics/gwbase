# Step 2: Wells in Catchments

## Overview

Step 2 assigns each well to a terminal gage based on which upstream catchment the well falls within.

## Purpose

Link wells to terminal gages by spatial membership in upstream catchments. A well is assigned to a terminal gage if it falls within any of that gage's upstream catchments.

## Input Data

- Step 1 output: Terminal gage upstream catchment mapping
- Wells shapefile with geometry column
- Catchment shapefile with geometry

## Algorithm

1. Perform spatial join: wells within catchments
2. Map catchment IDs to terminal gage IDs using Step 1 output
3. Assign each well to its terminal gage(s)

## Output

**File:** `outputs/interim/step02_wells_in_catchments.parquet`

**Columns:**
- `well_id`: Well identifier
- `gage_id`: Terminal gage identifier
- `gage_name`: Gage name
- `catchment_id`: Catchment ID containing the well

