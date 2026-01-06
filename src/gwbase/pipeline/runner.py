"""Pipeline execution runner."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from gwbase.config.schema import Config
from gwbase.pipeline.steps import (
    step01, step02, step03, step04, step05,
    step06, step07, step08, step09
)

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates execution of pipeline steps."""
    
    def __init__(self, config: Config):
        """
        Initialize pipeline runner.
        
        Parameters:
        -----------
        config : Config
            Configuration object
        """
        self.config = config
        self.step_outputs: Dict[int, Dict[str, Any]] = {}
        
        # Ensure output directories exist
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "interim").mkdir(exist_ok=True)
        (self.config.output_dir / "summaries").mkdir(exist_ok=True)
    
    def run_step(self, step_num: int, force: bool = False) -> Dict[str, Any]:
        """
        Run a specific pipeline step.
        
        Parameters:
        -----------
        step_num : int
            Step number (1-9)
        force : bool
            If True, rerun even if outputs exist
            
        Returns:
        --------
        dict
            Step output summary
        """
        logger.info(f"=" * 60)
        logger.info(f"Running Step {step_num}")
        logger.info(f"=" * 60)
        
        step_funcs = {
            1: step01.run_step01,
            2: step02.run_step02,
            3: step03.run_step03,
            4: step04.run_step04,
            5: step05.run_step05,
            6: step06.run_step06,
            7: step07.run_step07,
            8: step08.run_step08,
            9: step09.run_step09,
        }
        
        if step_num not in step_funcs:
            raise ValueError(f"Invalid step number: {step_num}. Must be 1-9.")
        
        # Check if output exists using summary file
        summary_path = self.config.get_summary_path(step_num)
        
        if not force and summary_path.exists():
            logger.info(f"Step {step_num} output already exists at {summary_path}. Use force=True to rerun.")
            import json
            with open(summary_path) as f:
                summary = json.load(f)
                self.step_outputs[step_num] = summary
                return summary
        
        # Run step
        try:
            step_func = step_funcs[step_num]
            summary = step_func(self.config, self.step_outputs)
            self.step_outputs[step_num] = summary
            return summary
        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}", exc_info=True)
            raise
    
    def run_all(self, start_from: int = 1) -> Dict[int, Dict[str, Any]]:
        """
        Run all pipeline steps in sequence.
        
        Parameters:
        -----------
        start_from : int
            Start from this step number (for resuming)
            
        Returns:
        --------
        dict
            All step outputs keyed by step number
        """
        logger.info("Starting complete pipeline execution")
        
        for step_num in range(start_from, 10):
            try:
                self.run_step(step_num)
            except Exception as e:
                logger.error(f"Pipeline stopped at step {step_num}: {e}")
                raise
        
        logger.info("Pipeline execution complete")
        return self.step_outputs

