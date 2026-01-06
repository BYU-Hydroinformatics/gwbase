"""Step 1: Identify terminal gages and delineate upstream catchments."""
import logging
from gwbase.config.schema import Config
from gwbase.io.readers import load_hydrography_data
from gwbase.io.writers import save_step_output, save_summary
from gwbase.io.paths import get_interim_path, get_summary_path
from gwbase.geo.terminal_gage import identify_terminal_gages

logger = logging.getLogger(__name__)


def run_step01(config: Config, previous_outputs: dict) -> dict:
    """Execute Step 1: Terminal gage identification."""
    logger.info("Step 1: Identifying terminal gages and upstream catchments")
    
    # Load spatial data
    gages_gdf, streams_gdf, subbasin_gdf, wells_gdf = load_hydrography_data(
        config.data_paths.gages_csv,
        config.data_paths.streams_shp,
        config.data_paths.subbasin_shp,
        config.data_paths.wells_shp
    )
    
    # Identify terminal gages
    terminal_gages_df = identify_terminal_gages(
        gages_gdf=gages_gdf,
        stream_gdf=streams_gdf,
        subbasin_gdf=subbasin_gdf,
        manual_remove=config.processing.terminal_gages.manual_remove,
        manual_add=config.processing.terminal_gages.manual_add
    )
    
    # Save output
    output_path = get_interim_path(config.output_dir, 1)
    save_step_output(terminal_gages_df, output_path)
    
    # Generate summary
    summary = {
        'step': 1,
        'output_file': str(output_path),
        'n_terminal_gages': int(terminal_gages_df['Gage_ID'].nunique()),
        'n_total_catchments': int(terminal_gages_df['Upstream_Catchment_ID'].nunique()),
        'terminal_gage_ids': terminal_gages_df['Gage_ID'].unique().tolist(),
    }
    
    summary_path = get_summary_path(config.output_dir, 1)
    save_summary(summary, summary_path)
    
    logger.info(f"Step 1 complete: {summary['n_terminal_gages']} terminal gages identified")
    return summary

