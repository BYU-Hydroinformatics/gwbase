# GWBASE Documentation

**Groundwater-Baseflow Analysis System**

GWBASE is an algorithm for assessing the impact of groundwater decline on baseflow in US streams. It provides a systematic workflow for linking groundwater level changes to streamflow variations under baseflow-dominated conditions.

## Overview

Baseflow—the portion of streamflow derived from groundwater discharge—is critical for maintaining perennial streamflow, supporting aquatic ecosystems, and sustaining water supplies during dry periods. GWBASE quantifies the relationship between changes in water table elevation (ΔWTE) and changes in streamflow (ΔQ) to assess how groundwater decline affects baseflow.

## Key Features

- **9-step analytical workflow** from raw data to statistical analysis
- **Spatial analysis** linking wells to stream networks and gages
- **PCHIP interpolation** for temporal alignment of groundwater data
- **Elevation-based filtering** to identify hydraulically connected wells
- **Baseflow-dominated period detection** using machine learning classification
- **Delta metrics computation** (ΔWTE, ΔQ) with lag analysis
- **Statistical analysis** including linear regression, mutual information, and cross-correlation

## Documentation Contents

- [Installation](installation.md) - Setting up GWBASE
- [Process Overview](overview.md) - Understanding the 9-step GWBASE workflow
- [Code Organization](code-organization.md) - Structure of the gwbase package
- [Data Guide](data-guide.md) - Preparing and organizing input data
- [Running the Code](running-the-code.md) - Executing the workflow
- [API Reference](api-reference.md) - Function documentation

## Quick Start

```python
import gwbase

# Load your data
stream_gdf = gwbase.load_hydrography_data('data/raw/streams.shp')
wells_gdf, well_ts, well_info = gwbase.load_groundwater_data('data/raw/groundwater/')

# Run interpolation
daily_data = gwbase.interpolate_daily_pchip(well_ts)

# Compute delta metrics
data_with_deltas = gwbase.compute_delta_metrics(paired_data)

# Analyze relationships
gage_stats = gwbase.compute_regression_by_gage(data_with_deltas)
```

## Citation

If you use GWBASE in your research, please cite:

> Li, X., Jones, N.L., Williams, G.P., Aghababaei, A., & Hales, R.C. (2024). GWBASE – An Algorithm for Assessing the Impact of Groundwater Decline on Baseflow in US Streams.

## License

GWBASE is released under the MIT License.
