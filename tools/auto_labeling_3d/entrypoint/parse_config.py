from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from mmengine import Config


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for logging."""

    level: str
    work_dir: Path

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Path) -> LoggingConfig:
        return cls(
            level=data.get("level", "INFO"),
            work_dir=Path(data.get("work_dir", base_dir / "work_dirs")),
        )


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for model checkpoint."""

    model_zoo_url: str
    checkpoint_path: Path


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model."""

    name: str
    model_config: Path
    checkpoint: CheckpointConfig


@dataclass(frozen=True)
class CreateInfoConfig:
    """Configuration for create_info step."""

    output_dir: Path
    model_list: List[ModelConfig]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CreateInfoConfig:
        model_list = []
        for model in data["model_list"]:
            checkpoint = CheckpointConfig(
                model_zoo_url=model["checkpoint"]["model_zoo_url"],
                checkpoint_path=Path(model["checkpoint"]["checkpoint_path"]),
            )
            model_cfg = ModelConfig(
                name=model["name"], model_config=Path(model["model_config"]), checkpoint=checkpoint
            )
            model_list.append(model_cfg)

        return cls(output_dir=Path(data["output_dir"]), model_list=model_list)


@dataclass(frozen=True)
class EnsembleInfosConfig:
    """Configuration for ensemble step."""

    config: Path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnsembleInfosConfig:
        return cls(config=Path(data["config"]))


@dataclass(frozen=True)
class CreatePseudoT4datasetConfig:
    """Configuration for pseudo_dataset step."""

    config: Path
    overwrite: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CreatePseudoT4datasetConfig:
        return cls(config=Path(data["config"]), overwrite=data["overwrite"])


@dataclass(frozen=True)
class ChangeDirectoryStructureConfig:
    """Configuration for change_directory_structure step."""

    version_dir_name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChangeDirectoryStructureConfig:
        return cls(version_dir_name=data["version_dir_name"])


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration."""

    logging: LoggingConfig
    root_path: Path
    create_info: Optional[CreateInfoConfig]
    ensemble_infos: Optional[EnsembleInfosConfig]
    create_pseudo_t4dataset: Optional[CreatePseudoT4datasetConfig]
    change_directory_structure: Optional[ChangeDirectoryStructureConfig]

    @classmethod
    def from_file(cls, config_path: Path) -> "PipelineConfig":
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

        with config_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        if not isinstance(yaml_data, dict):
            raise TypeError("Top-level configuration must be a mapping")

        # Parse logging config
        logging_cfg = LoggingConfig.from_dict(data=yaml_data["logging"], base_dir=config_path.parent)

        # Parse root_path
        root_path = Path(yaml_data["root_path"])

        # Parse create_info config
        create_info_cfg = None
        if "create_info" in yaml_data:
            create_info_cfg = CreateInfoConfig.from_dict(data=yaml_data["create_info"])

        # Parse ensemble config
        ensemble_infos_cfg = None
        if "ensemble_infos" in yaml_data:
            ensemble_infos_cfg = EnsembleInfosConfig.from_dict(data=yaml_data["ensemble_infos"])

        # Parse pseudo_dataset config
        create_pseudo_t4dataset_cfg = None
        if "create_pseudo_t4dataset" in yaml_data:
            create_pseudo_t4dataset_cfg = CreatePseudoT4datasetConfig.from_dict(
                data=yaml_data["create_pseudo_t4dataset"]
            )

        # Parse change_directory_structure config
        change_directory_structure_cfg = None
        if "change_directory_structure" in yaml_data:
            change_directory_structure_cfg = ChangeDirectoryStructureConfig.from_dict(
                data=yaml_data["change_directory_structure"]
            )

        return cls(
            logging=logging_cfg,
            root_path=root_path,
            create_info=create_info_cfg,
            ensemble_infos=ensemble_infos_cfg,
            create_pseudo_t4dataset=create_pseudo_t4dataset_cfg,
            change_directory_structure=change_directory_structure_cfg,
        )


def load_model_config(model: ModelConfig, work_dir: Path) -> Config:
    """
    Load mmengine Config for a specific model.

    Args:
        model (ModelConfig): Model configuration containing path to model config file.
        work_dir (Path): Working directory for the pipeline.

    Returns:
        Config: Loaded mmengine Config object.
    """
    cfg = Config.fromfile(str(model.model_config))
    cfg.work_dir = str(work_dir / model.name)
    return cfg


def load_ensemble_config(config_path: Path) -> Config:
    """
    Load ensemble configuration file.

    Args:
        config_path (Path): Path to the ensemble configuration file.

    Returns:
        Config: Loaded mmengine Config object.
    """
    return Config.fromfile(str(config_path))


def load_t4dataset_config(config_path: Path) -> Config:
    """
    Load T4dataset configuration file.

    Args:
        config_path (Path): Path to the T4dataset configuration file.

    Returns:
        Config: Loaded mmengine Config object.
    """
    return Config.fromfile(str(config_path))
