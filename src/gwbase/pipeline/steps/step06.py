"""Step 6: Elevation-based filtering using vertical buffer threshold."""
import logging
import pandas as pd
from gwbase.config.schema import Config
from gwbase.io.readers import load_step_output
from gwbase.io.writers import save_step_output, save_summary
from gwbase.io.paths import get_interim_path, get_summary_path
from gwbase.hydro.filters import filter_by_elevation_buffer

logger = logging.getLogger(__name__)


def run_step06(config: Config, previous_outputs: dict) -> dict:
    """Execute Step 6: Filter wells by elevation buffer."""
    logger.info("Step 6: Filtering wells by elevation buffer")
    
    # Load Step 3 output (well-reach links)
    step03_path = get_interim_path(config.output_dir, 3)
    well_reach_df = load_step_output(step03_path)
    
    # Load Step 5 output (daily interpolated WTE)
    step05_path = get_interim_path(config.output_dir, 5)
    daily_wte_df = load_step_output(step05_path)
    
    # Merge daily WTE with well-reach data
    merged_daily = pd.merge(
        daily_wte_df,
        well_reach_df[['well_id', 'gage_id', 'reach_id', 'reach_elevation_m']],
        on='well_id',
        how='inner'
    )
    
    # Filter by elevation buffer
    connected_wells_df = filter_by_elevation_buffer(
        well_data=merged_daily,
        reach_data=well_reach_df[['well_id', 'reach_elevation_m']],
        vertical_buffer_meters=config.processing.elevation_filter.vertical_buffer_meters,
        wte_units_feet=config.processing.elevation_filter.wte_units_feet,
        conversion_factor=config.processing.elevation_filter.conversion_factor
    )
    
    # Save output
    output_path = get_interim_path(config.output_dir, 6)
    save_step_output(connected_wells_df, output_path)
    
    # Generate summary
    summary = {
        'step': 6,
        'output_file': str(output_path),
        'n_wells_before': int(merged_daily['well_id'].nunique()),
        'n_wells_after': int(connected_wells_df['well_id'].nunique()),
        'wells_removed': int(merged_daily['well_id'].nunique()) - int(connected_wells_df['well_id'].nunique()),
    }
    
    summary_path = get_summary_path(config.output_dir, 6)
    save_summary(summary, summary_path)
    
    logger.info(f"Step 6 complete: {summary['n_wells_after']} connected wells retained")
    return summary

