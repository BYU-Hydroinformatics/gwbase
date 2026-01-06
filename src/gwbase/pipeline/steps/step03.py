"""Step 3: Associate each well with nearest stream segment and record reach elevation."""
import logging
import pandas as pd
from gwbase.config.schema import Config
from gwbase.io.readers import load_hydrography_data, load_step_output
from gwbase.io.writers import save_step_output, save_summary
from gwbase.io.paths import get_interim_path, get_summary_path
from gwbase.geo.well_reach import find_nearest_reach_and_elevation

logger = logging.getLogger(__name__)


def run_step03(config: Config, previous_outputs: dict) -> dict:
    """Execute Step 3: Find nearest reach for each well."""
    logger.info("Step 3: Associating wells with nearest stream reaches")
    
    # Load Step 2 output
    step02_path = get_interim_path(config.output_dir, 2)
    wells_in_catchments_df = load_step_output(step02_path)
    
    # Load spatial data
    gages_gdf, streams_gdf, subbasin_gdf, wells_gdf = load_hydrography_data(
        config.data_paths.gages_csv,
        config.data_paths.streams_shp,
        config.data_paths.subbasin_shp,
        config.data_paths.wells_shp
    )
    
    # Load reach elevation data if available
    reach_elevation_df = None
    if config.data_paths.reach_elevation_csv and config.data_paths.reach_elevation_csv.exists():
        logger.info(f"Loading reach elevation data from {config.data_paths.reach_elevation_csv}")
        reach_elevation_df = pd.read_csv(config.data_paths.reach_elevation_csv)
    
    # Find nearest reach for each well
    well_reach_links_df = find_nearest_reach_and_elevation(
        wells_gdf=wells_gdf,
        streams_gdf=streams_gdf,
        reach_elevation_df=reach_elevation_df
    )
    
    # Merge with well-gage assignments from Step 2
    well_reach_df = pd.merge(
        wells_in_catchments_df,
        well_reach_links_df,
        on='well_id',
        how='inner'
    )
    
    # Save output
    output_path = get_interim_path(config.output_dir, 3)
    save_step_output(well_reach_df, output_path)
    
    # Generate summary
    summary = {
        'step': 3,
        'output_file': str(output_path),
        'n_wells': int(well_reach_df['well_id'].nunique()),
        'n_reaches': int(well_reach_df['reach_id'].nunique()),
        'n_wells_with_elevation': int(well_reach_df['reach_elevation_m'].notna().sum()),
    }
    
    summary_path = get_summary_path(config.output_dir, 3)
    save_summary(summary, summary_path)
    
    logger.info(f"Step 3 complete: {summary['n_wells']} wells associated with reaches")
    return summary

