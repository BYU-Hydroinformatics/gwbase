"""Configuration schema using Pydantic."""
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import yaml


class DataPaths(BaseModel):
    """Paths to input data files."""
    groundwater_csv: Path
    streamflow_dir: Path
    subbasin_shp: Path
    streams_shp: Path
    gages_csv: Path
    wells_shp: Path
    reach_elevation_csv: Optional[Path] = None
    
    @field_validator('*', mode='before')
    @classmethod
    def convert_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class TerminalGageConfig(BaseModel):
    """Terminal gage identification parameters."""
    manual_remove: List[int] = Field(default_factory=list)
    manual_add: List[int] = Field(default_factory=list)


class WellFilteringConfig(BaseModel):
    """Well filtering parameters."""
    min_observations: int = Field(default=5, ge=1)
    min_time_span_days: int = Field(default=730, ge=1)  # 2 years
    z_score_threshold: float = Field(default=3.0, ge=0)
    iqr_multiplier: float = Field(default=1.5, ge=0)


class ElevationFilterConfig(BaseModel):
    """Elevation-based filtering parameters."""
    vertical_buffer_meters: float = Field(default=30.0, ge=0)
    wte_units_feet: bool = Field(default=True)  # Convert from feet if True
    conversion_factor: float = Field(default=0.3048)  # feet to meters


class InterpolationConfig(BaseModel):
    """PCHIP interpolation settings."""
    target_frequency: str = Field(default="D")  # Daily
    extrapolate: bool = Field(default=False)
    max_gap_days: int = Field(default=365, ge=1)


class BFDConfig(BaseModel):
    """Baseflow-dominated filtering."""
    bfd_column: str = Field(default="ML_BFD")
    bfd_value: int = Field(default=1)


class DeltaMetricsConfig(BaseModel):
    """Delta metrics calculation."""
    baseline_method: str = Field(default="first_bfd")  # Options: "first_bfd", "mean", "median"
    delta_wte_column: str = Field(default="delta_wte")
    delta_q_column: str = Field(default="delta_q")


class AnalysisConfig(BaseModel):
    """Step 9 analysis parameters."""
    correlation_methods: List[str] = Field(default_factory=lambda: ["pearson", "spearman"])
    mi_n_bins: int = Field(default=10, ge=2)
    ccf_max_lag_days: int = Field(default=3650, ge=1)  # 10 years
    min_pairs_for_analysis: int = Field(default=10, ge=1)


class ProcessingConfig(BaseModel):
    """General processing parameters."""
    terminal_gages: TerminalGageConfig = Field(default_factory=TerminalGageConfig)
    well_filtering: WellFilteringConfig = Field(default_factory=WellFilteringConfig)
    elevation_filter: ElevationFilterConfig = Field(default_factory=ElevationFilterConfig)
    interpolation: InterpolationConfig = Field(default_factory=InterpolationConfig)
    bfd: BFDConfig = Field(default_factory=BFDConfig)
    delta_metrics: DeltaMetricsConfig = Field(default_factory=DeltaMetricsConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


class Config(BaseModel):
    """Main configuration model."""
    data_paths: DataPaths
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output_dir: Path = Field(default=Path("./outputs"))
    
    @field_validator('output_dir', mode='before')
    @classmethod
    def convert_output_dir(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.model_dump(mode='python'), f, default_flow_style=False)
    
    def get_interim_path(self, step_num: int) -> Path:
        """Get path to interim output file for a step."""
        from gwbase.io.paths import get_interim_path
        return get_interim_path(self.output_dir, step_num)
    
    def get_summary_path(self, step_num: int) -> Path:
        """Get path to step summary file."""
        from gwbase.io.paths import get_summary_path
        return get_summary_path(self.output_dir, step_num)
