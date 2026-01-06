# Notebook to Pipeline Step Mapping

This document maps each notebook in the `notebooks/` directory to the corresponding pipeline steps and core modules in the `gwbase` package.

## Overview

The `gwbase` package implements a 9-step workflow based on the paper methodology. The original notebooks contain exploratory analysis, visualization, and processing logic. This mapping shows where notebook code should be extracted and how it maps to the package structure.

## Mapping Table

| Notebook | Pipeline Step(s) | Core Module(s) | Notes |
|----------|-----------------|----------------|-------|
| `01_data_inventory.ipynb` | Step 4 (part) | `gwbase.hydro.qc` | Data quality assessment and outlier detection |
| `02_terminal_gages.ipynb` | Step 1 | `gwbase.geo.terminal_gage` | Terminal gage identification and upstream catchment mapping |
| `03_pchip_interpolation.ipynb` | Step 5 | `gwbase.hydro.interp` | PCHIP daily interpolation |
| `04_pairing_wells_gages.ipynb` | Steps 2, 3, 6, 7 | `gwbase.geo.well_catchment`, `gwbase.geo.well_reach`, `gwbase.hydro.filters`, `gwbase.hydro.pairing` | Well-catchment assignment, reach pairing, elevation filtering, BFD pairing |
| `05_delta_metrics.ipynb` | Step 8, Step 9 (part) | `gwbase.hydro.deltas`, `gwbase.metrics.linear` | Delta metric calculation, correlation analysis |
| `06_plot_per_gage.ipynb` | Visualization | `gwbase.viz.plots` | Time series plotting (not a pipeline step) |
| `07_mapping.ipynb` | Visualization | `gwbase.viz.maps` | Spatial visualization (not a pipeline step) |
| `08_mi.ipynb` | Step 9 (part) | `gwbase.metrics.mi`, `gwbase.metrics.ccf` | Mutual information and cross-correlation analysis |

## Detailed Mapping

### Step 1: Terminal Gages

**Source Notebook:** `02_terminal_gages.ipynb`

**Core Module:** `gwbase.geo.terminal_gage`

**Extracted Functions:**
- `identify_terminal_gages()` - Main terminal gage identification logic

### Step 2: Wells in Catchments

**Source Notebook:** `04_pairing_wells_gages.ipynb` (Buffer1 section)

**Core Module:** `gwbase.geo.well_catchment`

**Extracted Functions:**
- `assign_wells_to_upstream_catchments()` - Spatial join of wells to catchments

### Step 3: Well-Reach Association

**Source Notebook:** `01_data_inventory.ipynb`, `04_pairing_wells_gages.ipynb`

**Core Module:** `gwbase.geo.well_reach`

**Extracted Functions:**
- `find_nearest_reach_and_elevation()` - Nearest reach search and elevation lookup

### Step 5: PCHIP Interpolation

**Source Notebook:** `03_pchip_interpolation.ipynb`

**Core Module:** `gwbase.hydro.interp`

**Extracted Functions:**
- `pchip_daily_interpolation()` - PCHIP interpolation to daily frequency

### Step 9: Relationship Analysis

**Source Notebook:** `05_delta_metrics.ipynb`, `08_mi.ipynb`

**Core Modules:** `gwbase.metrics.linear`, `gwbase.metrics.mi`, `gwbase.metrics.ccf`

**Extracted Functions (TODO):**
- Correlation analysis (Pearson, Spearman)
- Mutual information calculation
- Cross-correlation function (CCF) with lag analysis

