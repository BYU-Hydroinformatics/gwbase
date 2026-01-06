"""Command-line interface using Typer."""
import typer
from pathlib import Path
from typing import Optional
import logging
import sys
from gwbase.config.schema import Config
from gwbase.pipeline.runner import PipelineRunner

app = typer.Typer(
    name="gwbase",
    help="Groundwater-Surface Water Interaction Analysis Pipeline",
    add_completion=False
)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True
    ),
    step: Optional[int] = typer.Option(
        None,
        "--step", "-s",
        help="Run specific step number (1-9). If omitted, runs all steps.",
        min=1,
        max=9
    ),
    start_from: int = typer.Option(
        1,
        "--start-from",
        help="Start pipeline from this step number",
        min=1,
        max=9
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rerun steps even if outputs exist"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """Run the pipeline or a specific step."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        cfg = Config.from_yaml(config)
        logger.info(f"Loaded configuration from {config}")
        
        runner = PipelineRunner(cfg)
        
        if step:
            logger.info(f"Running step {step} only")
            runner.run_step(step, force=force)
        else:
            logger.info(f"Running complete pipeline starting from step {start_from}")
            runner.run_all(start_from=start_from)
        
        typer.echo("✅ Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        typer.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@app.command()
def validate(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to configuration YAML file",
        exists=True
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """Validate configuration and check input data availability."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        cfg = Config.from_yaml(config)
        logger.info("Configuration file is valid")
        
        issues = []
        if not cfg.data_paths.groundwater_csv.exists():
            issues.append(f"Groundwater CSV not found: {cfg.data_paths.groundwater_csv}")
        if not cfg.data_paths.streamflow_dir.exists():
            issues.append(f"Streamflow directory not found: {cfg.data_paths.streamflow_dir}")
        if not cfg.data_paths.subbasin_shp.exists():
            issues.append(f"Subbasin shapefile not found: {cfg.data_paths.subbasin_shp}")
        if not cfg.data_paths.streams_shp.exists():
            issues.append(f"Streams shapefile not found: {cfg.data_paths.streams_shp}")
        if not cfg.data_paths.gages_csv.exists():
            issues.append(f"Gages CSV not found: {cfg.data_paths.gages_csv}")
        if not cfg.data_paths.wells_shp.exists():
            issues.append(f"Wells shapefile not found: {cfg.data_paths.wells_shp}")
        
        if issues:
            for issue in issues:
                typer.echo(f"⚠️  {issue}", err=True)
            sys.exit(1)
        else:
            typer.echo("✅ All input data paths are valid")
            
    except Exception as e:
        typer.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for console script."""
    app()


if __name__ == "__main__":
    main()

