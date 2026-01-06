"""Step 7: Pair daily groundwater and streamflow only on baseflow-dominated days."""
import logging
from gwbase.config.schema import Config
from gwbase.io.readers import load_streamflow_data, load_step_output
from gwbase.io.writers import save_step_output, save_summary
from gwbase.io.paths import get_interim_path, get_summary_path
from gwbase.hydro.pairing import pair_wte_with_streamflow_bfd

logger = logging.getLogger(__name__)


def run_step07(config: Config, previous_outputs: dict) -> dict:
    """Execute Step 7: Pair WTE with streamflow on BFD days."""
    logger.info("Step 7: Pairing WTE with streamflow on BFD days")
    
    # Load Step 6 output (connected wells with daily WTE)
    step06_path = get_interim_path(config.output_dir, 6)
    connected_wells_df = load_step_output(step06_path)
    
    # Load streamflow data
    streamflow_df = load_streamflow_data(
        streamflow_dir=config.data_paths.streamflow_dir,
        bfd_column=config.processing.bfd.bfd_column,
        bfd_value=config.processing.bfd.bfd_value
    )
    
    # Pair WTE with streamflow on BFD days
    bfd_pairs_df = pair_wte_with_streamflow_bfd(
        wte_data=connected_wells_df,
        streamflow_data=streamflow_df,
        bfd_value=config.processing.bfd.bfd_value
    )
    
    # Save output
    output_path = get_interim_path(config.output_dir, 7)
    save_step_output(bfd_pairs_df, output_path)
    
    # Generate summary
    summary = {
        'step': 7,
        'output_file': str(output_path),
        'n_wells': int(bfd_pairs_df['well_id'].nunique()),
        'n_gages': int(bfd_pairs_df['gage_id'].nunique()),
        'n_pairs': len(bfd_pairs_df),
    }
    
    summary_path = get_summary_path(config.output_dir, 7)
    save_summary(summary, summary_path)
    
    logger.info(f"Step 7 complete: {summary['n_pairs']:,} WTE-Q pairs")
    return summary

