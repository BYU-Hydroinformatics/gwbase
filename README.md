# GWBASE: Groundwater-Surface Water Interaction Analysis Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python library for analyzing groundwater-surface water interactions using time series correlation, spatial analysis, and statistical methods.

## Features

- **Data Processing**: Load, clean, and interpolate groundwater and streamflow time series
- **Spatial Analysis**: Terminal gage identification, watershed delineation, and well-gage pairing
- **Correlation Analysis**: Cross-correlation functions, lag analysis, and statistical testing
- **Mutual Information**: Non-linear relationship detection between groundwater and surface water
- **Visualization**: Time series plots, scatter plots, correlation maps, and statistical summaries
- **Mapping**: Comprehensive spatial visualizations and watershed mapping capabilities

## Installation

### From PyPI (recommended)
```bash
pip install gwbase
```

### From Source
```bash
git clone https://github.com/BYU-Hydroinformatics/gwbase.git
cd gwbase
pip install -e .
```

### With Optional Dependencies
```bash
# For enhanced mapping capabilities
pip install gwbase[maps]

# For development
pip install gwbase[dev]

# For all features
pip install gwbase[full]
```

## Quick Start

### Basic Usage

```python
import gwbase as gw
import pandas as pd

# Load data
well_data = gw.load_groundwater_data('path/to/groundwater.csv')
streamflow_data = gw.load_streamflow_data('path/to/streamflow/')

# Interpolate daily values
interpolated_wells = gw.interpolate_daily_pchip(well_data)

# Calculate delta metrics (changes from baseline)
combined_data = gw.calculate_delta_metrics(combined_data)

# Analyze correlations
correlation_results = gw.analyze_correlation_patterns(combined_data)

# Create visualizations
gw.plot_correlation_scatter(combined_data, output_dir='./figures/')
```

### Complete Workflow

```python
from gwbase import GWBASEAnalysis

# Define data paths
data_paths = {
    'groundwater': 'data/groundwater_timeseries.csv',
    'streamflow_dir': 'data/streamflow/',
    'subbasin': 'data/catchments.shp',
    'streams': 'data/streams.shp',
    'gages': 'data/gages.csv',
    'wells': 'data/wells.shp'
}

# Initialize and run analysis
analysis = GWBASEAnalysis()
analysis.run_complete_analysis(data_paths)
```

## Examples

### Example 1: Basic Correlation Analysis
```python
# See examples/example_basic_analysis.py
python examples/example_basic_analysis.py
```

### Example 2: Complete Workflow
```python 
# See main_analysis.py for full workflow
python main_analysis.py
```

## Dependencies

**Core Requirements:**
- pandas >= 1.5.0
- numpy >= 1.21.0  
- scipy >= 1.9.0
- matplotlib >= 3.5.0
- geopandas >= 0.12.0
- networkx >= 2.8.0
- scikit-learn >= 1.1.0

**Optional Dependencies:**
- contextily >= 1.2.0 (enhanced mapping)
- cartopy >= 0.21.0 (projection support)
- geopy >= 2.3.0 (distance calculations)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.