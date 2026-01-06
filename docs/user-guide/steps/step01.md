# Step 1: Terminal Gage Identification

## Overview

Step 1 identifies terminal (downstream-most) gages in the stream network and maps all upstream catchments associated with each terminal gage.

## Purpose

Terminal gages serve as the reference points for upstream analysis. A terminal gage has **no other gage downstream** in the directed catchment network.

## Input Data

- Gages CSV file with columns: `id`, `name`, `latitude`, `longitude`
- Stream network shapefile with columns: `LINKNO`, `DSLINKNO`
- Catchment shapefile with column: `linkno` or `LINKNO`

## Algorithm

1. Build directed graph from stream network (LINKNO → DSLINKNO)
2. For each gage, check if any other gage is downstream
3. If no downstream gages exist, mark as terminal
4. For each terminal gage, find all catchments upstream (including terminal catchment)
5. Apply manual adjustments if specified in configuration

## Output

**File:** `outputs/interim/step01_terminal_gages.parquet`

**Columns:**
- `Gage_ID`: Terminal gage identifier
- `Gage_Name`: Gage name
- `Terminal_Catchment_ID`: Catchment ID of the terminal gage
- `Upstream_Catchment_ID`: ID of upstream catchment (includes terminal)

**Summary:** `outputs/summaries/step01_summary.json`

## Configuration

```yaml
processing:
  terminal_gages:
    manual_remove: []  # List of gage IDs to exclude
    manual_add: []     # List of gage IDs to include
```

