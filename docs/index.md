# GWBASE

A Scalable Framework to Quantify Groundwater–Baseflow Correlations Using Paired Well and Streamflow Observations

## Overview

This project analyzes correlations between groundwater levels (WTE) and baseflow using:

- **Groundwater data**: Well measurements (1900-2023)
- **Streamflow data**: USGS gage measurements 
- **Spatial data**: Basin boundaries, stream networks
- **Elevation data**: Stream reach elevations

## Workflow

1. **Data Inventory**: Quality assessment
2. **Terminal Gages**: Downstream gage identification
3. **PCHIP Interpolation**: Gap-filling for sparse data
4. **Well-Gage Pairing**: Spatial associations
5. **Delta Metrics**: Change-based correlation analysis
6. **Visualization**: Results and figures

## Quick Start

1. [Installation](installation.md) - Setup environment
2. [Data Overview](data-overview.md) - Understand datasets  
3. [Analysis Notebooks](notebooks.md) - Run workflow
4. [Results](results/overview.md) - View findings