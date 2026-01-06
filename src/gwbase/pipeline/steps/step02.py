"""Step 2: Locate wells within upstream catchments and link to terminal gages."""
import logging
from gwbase.config.schema import Config
from gwbase.io.readers import load_hydrography_data, load_step_output
from gwbase.io.writers import save_step_output, save_summary
from gwbase.io.paths import get_interim_path, get_summary_path
from gwbase.geo.well_catchment import assign_wells_to_upstream_catchments

logger = logging.getLogger(__name__)


def run_step02(config: Config, previous_outputs: dict) -> dict:
    """Execute Step 2: Assign wells to terminal gages based on catchment membership."""
    logger.info("Step 2: Locating wells within upstream catchments")
    
    # Load Step 1 output
    step01_path = get_interim_path(config.output_dir, 1)
    terminal_upstream_df = load_step_output(step01_path)
    
    # Load spatial data
    gages_gdf, streams_gdf, subbasin_gdf, wells_gdf = load_hydrography_data(
        config.data_paths.gages_csv,
        config.data_paths.streams_shp,
        config.data_paths.subbasin_shp,
        config.data_paths.wells_shp
    )
    
    # Assign wells to catchments
    wells_in_catchments_df = assign_wells_to_upstream_catchments(
        wells_gdf=wells_gdf,
        terminal_upstream_df=terminal_upstream_df,
        subbasin_gdf=subbasin_gdf
    )
    
    # Save output
    output_path = get_interim_path(config.output_dir, 2)
    save_step_output(wells_in_catchments_df, output_path)
    
    # Generate summary
    summary = {
        'step': 2,
        'output_file': str(output_path),
        'n_wells': int(wells_in_catchments_df['well_id'].nunique()),
        'n_gages': int(wells_in_catchments_df['gage_id'].nunique()),
        'n_well_gage_pairs': len(wells_in_catchments_df),
    }
    
    summary_path = get_summary_path(config.output_dir, 2)
    save_summary(summary, summary_path)
    
    logger.info(f"Step 2 complete: {summary['n_wells']} wells assigned")
    return summary

