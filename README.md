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
from pathlib import Path

# Load data
well_data = gw.load_groundwater_data(Path('path/to/groundwater.csv'))
streamflow_data = gw.load_streamflow_data(Path('path/to/streamflow/'))

# Interpolate daily values using PCHIP
interpolated_wells = gw.interpolate_daily_pchip(well_data)

# Calculate delta metrics (changes from baseline)
# Note: This requires paired data from the pipeline
delta_data = gw.calculate_delta_metrics(paired_data)
```

### Complete Workflow

The recommended way to run the complete analysis pipeline is using the command-line interface:

```bash
# Validate configuration
gwbase validate --config config.yaml

# Run complete pipeline
gwbase run --config config.yaml

# Run specific step
gwbase run --config config.yaml --step 5

# Resume from a step
gwbase run --config config.yaml --start-from 6
```

Or programmatically:

```python
from gwbase import Config, PipelineRunner
from pathlib import Path

# Load configuration
config = Config.from_yaml(Path('config.yaml'))

# Run pipeline
runner = PipelineRunner(config)
runner.run_all()
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