# Running the Code

This guide explains how to execute the GWBASE workflow, from running the complete pipeline to executing individual steps.

## Quick Start

### Option 1: Using the Main Script

The simplest way to run GWBASE is via the command-line interface:

```bash
python main_gwbase.py --data-dir /path/to/your/data
```

**Arguments:**
- `--data-dir`, `-d`: Base directory containing your data (default: current directory)
- `--steps`: Which steps to run (default: "all")
- `--buffer`: Elevation buffer in meters for Step 6 (default: 30)

### Option 2: Python Script

Create a custom Python script for more control:

```python
import gwbase
import os

# Your data directory
DATA_DIR = '/path/to/your/data'

# Run the workflow...
```

---

## Complete Workflow Example

Here's a complete example running all 9 steps:

```python
import gwbase
import os
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
FIGURES_DIR = 'reports/figures'

# Create output directories
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")

# Hydrography
stream_gdf = gwbase.load_hydrography_data(
    os.path.join(RAW_DIR, 'hydrography/streams.shp')
)
catchment_gdf = gwbase.load_hydrography_data(
    os.path.join(RAW_DIR, 'hydrography/catchments.shp')
)

# Groundwater (from CSV files)
wells_gdf, well_ts, well_info = gwbase.load_groundwater_data(
    well_locations_path=os.path.join(RAW_DIR, 'groundwater/wells.csv'),
    timeseries_path=os.path.join(RAW_DIR, 'groundwater/water_levels.csv')
)

# Streamflow
gage_info = gwbase.load_gage_info(
    os.path.join(RAW_DIR, 'streamflow/gage_info.csv')
)
streamflow = gwbase.load_streamflow_data(
    os.path.join(RAW_DIR, 'streamflow')
)

# BFD classification
bfd_class = gwbase.load_baseflow_classification(
    os.path.join(RAW_DIR, 'bfd/bfd_classification.csv')
)

# Reach elevations (may need to create this first)
reach_elev = gwbase.load_reach_elevations(
    os.path.join(PROCESSED_DIR, 'reach_elevations.csv')
)

# ============================================================
# STEP 1: Stream Network Analysis
# ============================================================
print("\n" + "="*60)
print("STEP 1: Building Stream Network")
print("="*60)

# Build network graph
G = gwbase.build_stream_network_graph(stream_gdf)

# Match gages to catchments
matched_gages = gwbase.match_gages_to_catchments(gage_info, catchment_gdf)

# Identify terminal gages
terminal_ids = gwbase.identify_terminal_gages(matched_gages, G)
terminal_gages = matched_gages[matched_gages['id'].isin(terminal_ids)]

# Delineate upstream catchments
upstream_df = gwbase.delineate_all_upstream_catchments(terminal_gages, G)

# Save
upstream_df.to_csv(os.path.join(PROCESSED_DIR, 'upstream_catchments.csv'), index=False)
print(f"Identified {len(terminal_ids)} terminal gages")

# ============================================================
# STEP 2: Locate Wells in Catchments
# ============================================================
print("\n" + "="*60)
print("STEP 2: Locating Wells")
print("="*60)

wells_with_gages = gwbase.locate_wells_in_catchments(
    wells_gdf, catchment_gdf, upstream_df
)
wells_with_gages.to_csv(os.path.join(PROCESSED_DIR, 'wells_in_catchments.csv'), index=False)

# ============================================================
# STEP 3: Associate Wells with Reaches
# ============================================================
print("\n" + "="*60)
print("STEP 3: Associating Wells with Reaches")
print("="*60)

well_reach = gwbase.associate_wells_with_reaches(
    wells_gdf, stream_gdf, gage_info, reach_elev
)
well_reach.to_csv(os.path.join(PROCESSED_DIR, 'well_reach.csv'), index=False)

# ============================================================
# STEP 4: Data Preprocessing
# ============================================================
print("\n" + "="*60)
print("STEP 4: Preprocessing")
print("="*60)

clean_data = gwbase.clean_well_data_for_interpolation(well_ts, min_points=5)
clean_data.to_csv(os.path.join(PROCESSED_DIR, 'well_ts_clean.csv'), index=False)

# ============================================================
# STEP 5: PCHIP Interpolation
# ============================================================
print("\n" + "="*60)
print("STEP 5: PCHIP Interpolation")
print("="*60)

daily_data = gwbase.interpolate_with_well_info(clean_data, well_info)
daily_data.to_csv(os.path.join(PROCESSED_DIR, 'well_pchip_daily.csv'), index=False)

# ============================================================
# STEP 6: Elevation Filtering
# ============================================================
print("\n" + "="*60)
print("STEP 6: Elevation Filtering")
print("="*60)

# Merge daily data with gage assignments
merged_data = gwbase.merge_well_reach_data(
    daily_data, well_reach, wells_with_gages, gage_info
)

# Apply elevation filter
filtered_data, elev_stats = gwbase.filter_by_elevation(
    merged_data, well_reach, distance_buffer_meters=30.0
)
filtered_data.to_csv(os.path.join(PROCESSED_DIR, 'filtered_by_elevation.csv'), index=False)

# ============================================================
# STEP 7: Pair with Streamflow
# ============================================================
print("\n" + "="*60)
print("STEP 7: Pairing with Streamflow")
print("="*60)

paired = gwbase.pair_wells_with_streamflow(
    filtered_data, streamflow, bfd_class
)
paired = gwbase.calculate_baseline_values(paired)
paired.to_csv(os.path.join(PROCESSED_DIR, 'paired_data.csv'), index=False)

# ============================================================
# STEP 8: Compute Delta Metrics
# ============================================================
print("\n" + "="*60)
print("STEP 8: Computing Delta Metrics")
print("="*60)

data_with_deltas = gwbase.compute_delta_metrics(paired)
data_with_deltas.to_csv(os.path.join(FEATURES_DIR, 'data_with_deltas.csv'), index=False)

# Create lag versions
lag_1yr = gwbase.create_lag_analysis(data_with_deltas, 1, 'years')
lag_1yr.to_csv(os.path.join(FEATURES_DIR, 'data_lag_1yr.csv'), index=False)

# ============================================================
# STEP 9: Analyze Relationships
# ============================================================
print("\n" + "="*60)
print("STEP 9: Analyzing Relationships")
print("="*60)

# Regression by gage
gage_stats = gwbase.compute_regression_by_gage(data_with_deltas)
gage_stats.to_csv(os.path.join(FEATURES_DIR, 'regression_by_gage.csv'), index=False)

# Regression by well
well_stats = gwbase.compute_regression_by_well(data_with_deltas)
well_stats.to_csv(os.path.join(FEATURES_DIR, 'regression_by_well.csv'), index=False)

# Mutual information
mi_results = gwbase.compute_mi_analysis(data_with_deltas)
mi_results.to_csv(os.path.join(FEATURES_DIR, 'mi_analysis.csv'), index=False)

# Summary
summary = gwbase.summarize_regression_results(gage_stats, well_stats)

# ============================================================
# CREATE VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("Creating Visualizations")
print("="*60)

gwbase.plot_regression_summary(
    gage_stats,
    os.path.join(FIGURES_DIR, 'regression')
)

gwbase.plot_delta_scatter(
    data_with_deltas,
    os.path.join(FIGURES_DIR, 'scatter_plots')
)

print("\n" + "="*60)
print("WORKFLOW COMPLETE")
print("="*60)
```

---

## Running Individual Steps

You can run steps individually if you have intermediate data saved:

### Starting from Step 5 (Interpolation)

```python
import gwbase
import pandas as pd

# Load cleaned data from Step 4
clean_data = pd.read_csv('data/processed/well_ts_clean.csv')
well_info = pd.read_csv('data/raw/groundwater/well_metadata.csv')

# Run Step 5
daily_data = gwbase.interpolate_daily_pchip(clean_data)
daily_data.to_csv('data/processed/well_pchip_daily.csv', index=False)
```

### Starting from Step 8 (Delta Metrics)

```python
import gwbase
import pandas as pd

# Load paired data from Step 7
paired = pd.read_csv('data/processed/paired_data.csv')

# Run Steps 8-9
data_with_deltas = gwbase.compute_delta_metrics(paired)
gage_stats = gwbase.compute_regression_by_gage(data_with_deltas)
```

---

## Running with Different Parameters

### Adjusting Elevation Buffer

```python
# Test different buffer values
for buffer in [10, 20, 30, 50, 100]:
    filtered, stats = gwbase.filter_by_elevation(
        merged_data, well_reach,
        distance_buffer_meters=buffer
    )
    print(f"Buffer {buffer}m: {len(filtered):,} records retained")

# Or use sensitivity analysis
sensitivity = gwbase.analyze_elevation_sensitivity(
    merged_data, well_reach,
    buffer_values=[10, 20, 30, 50, 100]
)
```

### Using Different Lag Periods

```python
# Create multiple lag versions
lags = [
    (3, 'months'),
    (6, 'months'),
    (1, 'years'),
    (2, 'years'),
]

for period, unit in lags:
    lag_data = gwbase.create_lag_analysis(data_with_deltas, period, unit)
    suffix = f'{period}yr' if unit == 'years' else f'{period}mo'
    lag_data.to_csv(f'data/features/data_lag_{suffix}.csv', index=False)
```

### Filtering to Top-Correlated Wells

```python
# Compute per-well statistics
well_stats = gwbase.compute_regression_by_well(data_with_deltas)

# Filter to top 10% by R²
top_wells_data = gwbase.filter_by_correlation(
    data_with_deltas,
    well_stats,
    percentile=10,
    correlation_col='r_squared'
)

# Re-run regression on filtered data
top_gage_stats = gwbase.compute_regression_by_gage(top_wells_data)
```

---

## Batch Processing Multiple Study Areas

```python
import gwbase
import os

study_areas = ['basin_a', 'basin_b', 'basin_c']

for area in study_areas:
    print(f"\n{'='*60}")
    print(f"Processing: {area}")
    print('='*60)

    data_dir = f'data/{area}'
    output_dir = f'results/{area}'
    os.makedirs(output_dir, exist_ok=True)

    # Load data for this area
    stream_gdf = gwbase.load_hydrography_data(
        os.path.join(data_dir, 'streams.shp')
    )
    # ... load other data ...

    # Run workflow
    # ... steps 1-9 ...

    # Save results
    gage_stats.to_csv(os.path.join(output_dir, 'results.csv'))

    print(f"Completed: {area}")
```

---

## Working with Large Datasets

For large datasets, consider these optimizations:

### Memory Management

```python
import pandas as pd
import gc

# Process in chunks by gage
gage_ids = data['gage_id'].unique()

all_results = []
for gage_id in gage_ids:
    gage_data = data[data['gage_id'] == gage_id]

    # Process this gage
    result = gwbase.compute_regression_by_gage(gage_data)
    all_results.append(result)

    # Free memory
    del gage_data
    gc.collect()

final_results = pd.concat(all_results)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
import gwbase

def process_gage(gage_data):
    """Process a single gage's data."""
    return gwbase.compute_regression_by_gage(gage_data)

# Split data by gage
gage_groups = [group for _, group in data.groupby('gage_id')]

# Process in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_gage, gage_groups))

final_results = pd.concat(results)
```

---

## Saving and Loading Intermediate Results

### Checkpointing Long Workflows

```python
import pickle

# Save intermediate state
checkpoint = {
    'graph': G,
    'terminal_gages': terminal_gages,
    'upstream_df': upstream_df,
}
with open('checkpoint_step1.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)

# Load checkpoint later
with open('checkpoint_step1.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
G = checkpoint['graph']
```

### Using Parquet for Large DataFrames

```python
# Faster read/write for large datasets
filtered_data.to_parquet('data/processed/filtered.parquet')
filtered_data = pd.read_parquet('data/processed/filtered.parquet')
```

---

## Error Handling

### Handling Missing Data

```python
import gwbase
import pandas as pd

# Check for required columns
required_cols = ['well_id', 'date', 'wte']
missing = [c for c in required_cols if c not in well_ts.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Handle missing values
well_ts = well_ts.dropna(subset=['wte'])
print(f"Records after removing NaN: {len(well_ts):,}")
```

### Graceful Failure Recovery

```python
import gwbase
import traceback

steps = [
    ('Step 1', lambda: gwbase.build_stream_network_graph(stream_gdf)),
    ('Step 2', lambda: gwbase.locate_wells_in_catchments(wells_gdf, catchment_gdf, upstream_df)),
    # ... more steps
]

for step_name, step_func in steps:
    try:
        print(f"Running {step_name}...")
        result = step_func()
        print(f"{step_name} completed successfully")
    except Exception as e:
        print(f"ERROR in {step_name}: {e}")
        traceback.print_exc()
        # Save what we have so far
        break
```

---

## Logging

### Enable Detailed Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gwbase_run.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use in workflow
logger.info("Starting GWBASE workflow")
logger.info(f"Processing {len(well_ts):,} well records")
```
