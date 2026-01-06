"""Step 4: Filter wells with insufficient observations and remove outliers."""
import logging
from gwbase.config.schema import Config
from gwbase.io.readers import load_groundwater_data
from gwbase.io.writers import save_step_output, save_summary
from gwbase.io.paths import get_interim_path, get_summary_path
from gwbase.hydro.qc import filter_wells_by_quality

logger = logging.getLogger(__name__)


def run_step04(config: Config, previous_outputs: dict) -> dict:
    """Execute Step 4: Filter wells by data quality."""
    logger.info("Step 4: Filtering wells by data quality")
    
    # Load raw groundwater data
    wte_data = load_groundwater_data(config.data_paths.groundwater_csv)
    
    # Filter wells by quality criteria
    filtered_wte_data = filter_wells_by_quality(
        well_ts=wte_data,
        min_observations=config.processing.well_filtering.min_observations,
        min_time_span_days=config.processing.well_filtering.min_time_span_days,
        z_score_threshold=config.processing.well_filtering.z_score_threshold,
        iqr_multiplier=config.processing.well_filtering.iqr_multiplier
    )
    
    # Save output
    output_path = get_interim_path(config.output_dir, 4)
    save_step_output(filtered_wte_data, output_path)
    
    # Generate summary
    summary = {
        'step': 4,
        'output_file': str(output_path),
        'n_wells_before': int(wte_data['well_id'].nunique()),
        'n_wells_after': int(filtered_wte_data['well_id'].nunique()),
        'wells_removed': int(wte_data['well_id'].nunique()) - int(filtered_wte_data['well_id'].nunique()),
    }
    
    summary_path = get_summary_path(config.output_dir, 4)
    save_summary(summary, summary_path)
    
    logger.info(f"Step 4 complete: {summary['n_wells_after']} wells retained")
    return summary

