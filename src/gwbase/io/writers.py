"""Data writing utilities."""
import pandas as pd
import json
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def save_step_output(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save step output as Parquet file.
    
    Parameters:
    -----------
    df : DataFrame
        Data to save
    output_path : Path
        Output file path (should have .parquet extension)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved step output to {output_path}")


def load_step_output(input_path: Path) -> pd.DataFrame:
    """
    Load step output from Parquet file.
    
    Parameters:
    -----------
    input_path : Path
        Input file path
        
    Returns:
    --------
    DataFrame
        Loaded data
    """
    logger.info(f"Loading step output from {input_path}")
    return pd.read_parquet(input_path)


def save_summary(summary: Dict[str, Any], summary_path: Path) -> None:
    """
    Save step summary as JSON file.
    
    Parameters:
    -----------
    summary : dict
        Summary dictionary (must be JSON-serializable)
    summary_path : Path
        Output file path
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Path objects to strings for JSON serialization
    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return convert_paths(obj.__dict__)
        return obj
    
    json_summary = convert_paths(summary)
    
    with open(summary_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    logger.info(f"Saved step summary to {summary_path}")
