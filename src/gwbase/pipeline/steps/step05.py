"""Step 5: Interpolate groundwater levels to daily using PCHIP."""
import logging
from gwbase.config.schema import Config
from gwbase.io.readers import load_step_output
from gwbase.io.writers import save_step_output, save_summary
from gwbase.io.paths import get_interim_path, get_summary_path
from gwbase.hydro.interp import pchip_daily_interpolation

logger = logging.getLogger(__name__)


def run_step05(config: Config, previous_outputs: dict) -> dict:
    """Execute Step 5: PCHIP daily interpolation."""
    logger.info("Step 5: Interpolating groundwater levels to daily using PCHIP")
    
    # Load Step 4 output (filtered wells)
    step04_path = get_interim_path(config.output_dir, 4)
    filtered_wte_data = load_step_output(step04_path)
    
    # Perform PCHIP interpolation
    daily_wte_data = pchip_daily_interpolation(
        well_ts=filtered_wte_data,
        target_frequency=config.processing.interpolation.target_frequency,
        extrapolate=config.processing.interpolation.extrapolate,
        max_gap_days=config.processing.interpolation.max_gap_days
    )
    
    # Save output
    output_path = get_interim_path(config.output_dir, 5)
    save_step_output(daily_wte_data, output_path)
    
    # Generate summary
    summary = {
        'step': 5,
        'output_file': str(output_path),
        'n_wells': int(daily_wte_data['well_id'].nunique()),
        'n_daily_records': len(daily_wte_data),
    }
    
    summary_path = get_summary_path(config.output_dir, 5)
    save_summary(summary, summary_path)
    
    logger.info(f"Step 5 complete: {summary['n_daily_records']:,} daily records")
    return summary

