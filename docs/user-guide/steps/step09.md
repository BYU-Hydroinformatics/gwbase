# Step 9: Relationship Analysis

## Overview

Step 9 analyzes relationships between ΔWTE and ΔQ and computes correlation metrics.

## Purpose

Quantify the strength and nature of groundwater-streamflow relationships using multiple methods.

## Input Data

- Step 8 output: Delta metrics

## Algorithm

1. **Linear Correlations:**
   - Pearson correlation coefficient (r)
   - Spearman rank correlation (ρ)
   - R² values
2. **Mutual Information:**
   - Non-linear relationship detection
   - Binned mutual information calculation
3. **Cross-Correlation Function (CCF):**
   - Lag analysis up to max_lag_days
   - Identify optimal lag time
4. Aggregate results per well-gage pair

## Output

**File:** `outputs/interim/step09_relationship_metrics.parquet`

## Status

**TODO:** Extract correlation, MI, and CCF functions from notebooks into `gwbase.metrics` modules.

