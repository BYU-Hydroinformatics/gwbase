# Step 8: Delta Metrics

## Overview

Step 8 computes delta metrics (changes from baseline) for both WTE and Q.

## Purpose

Quantify temporal changes relative to a baseline condition (typically first BFD day).

## Input Data

- Step 7 output: BFD pairs

## Algorithm

1. For each well-gage pair:
   - Identify baseline (first BFD day by default)
   - Record baseline WTE₀ and Q₀
2. Calculate deltas:
   - ΔWTE = WTE - WTE₀
   - ΔQ = Q - Q₀

## Output

**File:** `outputs/interim/step08_delta_metrics.parquet`

**Columns:**
- All columns from Step 7 plus:
- `delta_wte`: Change in WTE from baseline
- `delta_q`: Change in Q from baseline

