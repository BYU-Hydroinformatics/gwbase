# Quick Start Guide

This guide will help you get started with GWBASE quickly.

## Installation

### From Source

```bash
git clone https://github.com/your-username/gwbase.git
cd gwbase
pip install -e .
```

## Configuration

Create a configuration file `config.yaml` based on `src/gwbase/config/defaults.yaml`:

```yaml
data_paths:
  groundwater_csv: "data/raw/groundwater/groundwater_timeseries.csv"
  streamflow_dir: "data/raw/streamflow"
  subbasin_shp: "data/raw/hydrography/catchments.shp"
  streams_shp: "data/raw/hydrography/streams.shp"
  gages_csv: "data/raw/hydrography/gages.csv"
  wells_shp: "data/raw/hydrography/wells.shp"

processing:
  terminal_gages:
    manual_remove: []
    manual_add: []
  
  well_filtering:
    min_observations: 5
    min_time_span_days: 730
  
  elevation_filter:
    vertical_buffer_meters: 30.0

output_dir: "outputs"
```

## Basic Usage

### Validate Configuration

```bash
gwbase validate --config config.yaml
```

### Run Complete Pipeline

```bash
gwbase run --config config.yaml
```

### Run Specific Step

```bash
# Run step 5 (PCHIP interpolation) only
gwbase run --config config.yaml --step 5

# Force rerun even if output exists
gwbase run --config config.yaml --step 5 --force
```

### Resume from a Step

```bash
# Resume pipeline from step 6
gwbase run --config config.yaml --start-from 6
```

## Output Structure

Pipeline outputs are saved to the `output_dir` specified in your configuration:

```
outputs/
├── interim/
│   ├── step01_terminal_gages.parquet
│   ├── step02_wells_in_catchments.parquet
│   ├── step03_well_reach_links.parquet
│   ├── step04_filtered_wells.parquet
│   ├── step05_wte_daily_pchip.parquet
│   ├── step06_connected_wells.parquet
│   ├── step07_bfd_pairs.parquet
│   ├── step08_delta_metrics.parquet
│   └── step09_relationship_metrics.parquet
└── summaries/
    ├── step01_summary.json
    ├── step02_summary.json
    └── ...
```

## Python API

You can also use GWBASE programmatically:

```python
from gwbase.config.schema import Config
from gwbase.pipeline.runner import PipelineRunner

# Load configuration
config = Config.from_yaml("config.yaml")

# Run pipeline
runner = PipelineRunner(config)
outputs = runner.run_all()

# Run specific step
summary = runner.run_step(5)
```

