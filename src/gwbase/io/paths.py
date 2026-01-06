"""Standardized path management for pipeline outputs."""
from pathlib import Path
from typing import Optional


# Standardized interim filenames for each step
STEP_FILENAMES = {
    1: "step01_terminal_gages.parquet",
    2: "step02_wells_in_catchments.parquet",
    3: "step03_well_reach_links.parquet",
    4: "step04_filtered_wells.parquet",
    5: "step05_wte_daily_pchip.parquet",
    6: "step06_connected_wells.parquet",
    7: "step07_bfd_pairs.parquet",
    8: "step08_delta_metrics.parquet",
    9: "step09_relationship_metrics.parquet",
}


def get_interim_path(output_dir: Path, step_num: int) -> Path:
    """
    Get standardized interim output path for a pipeline step.
    
    Parameters:
    -----------
    output_dir : Path
        Base output directory
    step_num : int
        Step number (1-9)
        
    Returns:
    --------
    Path
        Full path to output file
        
    Raises:
    -------
    ValueError
        If step_num is not in range 1-9
    """
    if step_num not in STEP_FILENAMES:
        raise ValueError(f"Invalid step number: {step_num}. Must be 1-9.")
    
    interim_dir = output_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    return interim_dir / STEP_FILENAMES[step_num]


def get_summary_path(output_dir: Path, step_num: int) -> Path:
    """
    Get standardized summary output path.
    
    Parameters:
    -----------
    output_dir : Path
        Base output directory
    step_num : int
        Step number (1-9)
        
    Returns:
    --------
    Path
        Full path to summary JSON file
    """
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    return summaries_dir / f"step{step_num:02d}_summary.json"
