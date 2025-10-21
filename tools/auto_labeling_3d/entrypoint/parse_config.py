from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str
    work_dir: Path


@dataclass
class CheckpointConfig:
    """Configuration for model checkpoint."""
    model_zoo_url: str
    checkpoint_path: Path


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    model_config: Path
    checkpoint: CheckpointConfig


@dataclass
class CreateInfoConfig:
    """Configuration for create_info step."""
    root_path: Path
    output_dir: Path
    model_list: List[ModelConfig]


@dataclass
class EnsembleConfig:
    """Configuration for ensemble step."""
    config: Path


@dataclass
class TrackingConfig:
    """Configuration for tracking step."""
    input_path: Path
    output_path: Path


@dataclass
class PseudoDatasetConfig:
    """Configuration for pseudo_dataset step."""
    config: Path
    root_path: Path
    input_source: str
    overwrite: bool


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    logging: LoggingConfig
    create_info: CreateInfoConfig
    ensemble: EnsembleConfig
    tracking: TrackingConfig
    pseudo_dataset: PseudoDatasetConfig


def _parse_logging_config(logging_dict: Dict[str, Any], base_dir: Path) -> LoggingConfig:
    """Parse logging configuration from dictionary."""
    level = logging_dict.get("level", "INFO")
    work_dir = Path(logging_dict.get("work_dir", base_dir / "work_dirs"))
    return LoggingConfig(level=level, work_dir=work_dir)


def _parse_checkpoint(checkpoint_dict: Dict[str, Any]) -> CheckpointConfig:
    """Parse checkpoint configuration from dictionary."""
    model_zoo_url = checkpoint_dict.get("model_zoo_url", "")
    checkpoint_path = Path(checkpoint_dict.get("checkpoint_path", ""))
    return CheckpointConfig(model_zoo_url=model_zoo_url, checkpoint_path=checkpoint_path)


def _parse_model(model_dict: Dict[str, Any]) -> ModelConfig:
    """Parse model configuration from dictionary."""
    name = model_dict.get("name", "")
    model_config = Path(model_dict.get("model_config", ""))
    checkpoint_dict = model_dict.get("checkpoint", {})
    checkpoint = _parse_checkpoint(checkpoint_dict)
    return ModelConfig(name=name, model_config=model_config, checkpoint=checkpoint)


def _parse_create_info(create_info_dict: Optional[Dict[str, Any]], base_dir: Path) -> CreateInfoConfig:
    """Parse create_info configuration from dictionary."""
    if not create_info_dict:
        raise ValueError("create_info section is required in configuration")
    
    root_path = Path(create_info_dict.get("root_path", ""))
    output_dir = Path(create_info_dict.get("output_dir", ""))
    model_list_raw = create_info_dict.get("model_list", [])
    model_list = [_parse_model(m) for m in model_list_raw]
    
    return CreateInfoConfig(root_path=root_path, output_dir=output_dir, model_list=model_list)


def _parse_ensemble(ensemble_dict: Dict[str, Any], base_dir: Path) -> EnsembleConfig:
    """Parse ensemble configuration from dictionary."""
    config = Path(ensemble_dict.get("config", ""))
    return EnsembleConfig(config=config)


def _parse_tracking(tracking_dict: Optional[Dict[str, Any]], base_dir: Path) -> TrackingConfig:
    """Parse tracking configuration from dictionary."""
    if not tracking_dict:
        raise ValueError("tracking section is required in configuration")
    
    input_path = Path(tracking_dict.get("input_path", ""))
    output_path = Path(tracking_dict.get("output_path", ""))
    return TrackingConfig(input_path=input_path, output_path=output_path)


def _parse_pseudo_dataset(pseudo_dataset_dict: Dict[str, Any], base_dir: Path) -> PseudoDatasetConfig:
    """Parse pseudo_dataset configuration from dictionary."""
    config = Path(pseudo_dataset_dict.get("config", ""))
    root_path = Path(pseudo_dataset_dict.get("root_path", ""))
    input_source = pseudo_dataset_dict.get("input_source", "tracking")
    overwrite = pseudo_dataset_dict.get("overwrite", False)
    
    return PseudoDatasetConfig(
        config=config,
        root_path=root_path,
        input_source=input_source,
        overwrite=overwrite,
    )


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    """
    Load and parse pipeline configuration from YAML file.
    
    Args:
        config_path (Path): Path to the YAML configuration file.
        
    Returns:
        PipelineConfig: Parsed pipeline configuration.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        TypeError: If the configuration is not a valid mapping.
        ValueError: If required sections are missing.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        raw_config = yaml.safe_load(fp) or {}

    if not isinstance(raw_config, dict):
        raise TypeError("Top-level configuration must be a mapping")

    base_dir = config_path.parent

    logging_cfg = _parse_logging_config(raw_config.get("logging", {}), base_dir)
    create_info_cfg = _parse_create_info(raw_config.get("create_info"), base_dir)
    ensemble_cfg = _parse_ensemble(raw_config.get("ensemble", {}), base_dir)
    tracking_cfg = _parse_tracking(raw_config.get("tracking"), base_dir)
    pseudo_dataset_cfg = _parse_pseudo_dataset(raw_config.get("pseudo_dataset", {}), base_dir)

    return PipelineConfig(
        logging=logging_cfg,
        create_info=create_info_cfg,
        ensemble=ensemble_cfg,
        tracking=tracking_cfg,
        pseudo_dataset=pseudo_dataset_cfg,
    )
