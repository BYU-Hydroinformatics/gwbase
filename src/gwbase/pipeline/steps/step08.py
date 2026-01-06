"""Step 8: Compute delta metrics using first BFD day as baseline."""
import logging
from gwbase.config.schema import Config
from gwbase.io.readers import load_step_output
from gwbase.io.writers import save_step_output, save_summary
from gwbase.io.paths import get_interim_path, get_summary_path
from gwbase.hydro.deltas import calculate_delta_metrics

logger = logging.getLogger(__name__)


def run_step08(config: Config, previous_outputs: dict) -> dict:
    """Execute Step 8: Calculate delta metrics."""
    logger.info("Step 8: Computing delta metrics")
    
    # Load Step 7 output (BFD pairs)
    step07_path = get_interim_path(config.output_dir, 7)
    bfd_pairs_df = load_step_output(step07_path)
    
    # Calculate delta metrics
    delta_metrics_df = calculate_delta_metrics(
        paired_data=bfd_pairs_df,
        baseline_method=config.processing.delta_metrics.baseline_method,
        delta_wte_col=config.processing.delta_metrics.delta_wte_column,
        delta_q_col=config.processing.delta_metrics.delta_q_column
    )
    
    # Save output
    output_path = get_interim_path(config.output_dir, 8)
    save_step_output(delta_metrics_df, output_path)
    
    # Generate summary
    summary = {
        'step': 8,
        'output_file': str(output_path),
        'n_total_records': len(delta_metrics_df),
        'baseline_method': config.processing.delta_metrics.baseline_method,
    }
    
    summary_path = get_summary_path(config.output_dir, 8)
    save_summary(summary, summary_path)
    
    logger.info(f"Step 8 complete: Delta metrics calculated")
    return summary

