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
    def from_cfg(cls, logging_dict: Dict[str, Any], base_dir: Path) -> "LoggingConfig":
        """Parse logging configuration from dictionary."""
        level = logging_dict.get("level", "INFO")
        work_dir = Path(logging_dict.get("work_dir", base_dir / "work_dirs"))
        return cls(level=level, work_dir=work_dir)


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for model checkpoint."""

    model_zoo_url: str
    checkpoint_path: Path

    @classmethod
    def from_cfg(cls, checkpoint_dict: Dict[str, Any]) -> "CheckpointConfig":
        """Parse checkpoint configuration from dictionary."""
        model_zoo_url = checkpoint_dict.get("model_zoo_url", "")
        checkpoint_path = Path(checkpoint_dict.get("checkpoint_path", ""))
        return cls(model_zoo_url=model_zoo_url, checkpoint_path=checkpoint_path)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model."""

    name: str
    model_config: Path
    checkpoint: CheckpointConfig

    @classmethod
    def from_cfg(cls, model_dict: Dict[str, Any]) -> "ModelConfig":
        """Parse model configuration from dictionary."""
        name = model_dict.get("name", "")
        model_config = Path(model_dict.get("model_config", ""))
        checkpoint_dict = model_dict.get("checkpoint", {})
        checkpoint = CheckpointConfig.from_cfg(checkpoint_dict)
        return cls(name=name, model_config=model_config, checkpoint=checkpoint)


@dataclass(frozen=True)
class CreateInfoConfig:
    """Configuration for create_info step."""

    root_path: Path
    output_dir: Path
    model_list: List[ModelConfig]

    @classmethod
    def from_cfg(cls, create_info_dict: Optional[Dict[str, Any]], base_dir: Path) -> "CreateInfoConfig":
        """Parse create_info configuration from dictionary."""
        if not create_info_dict:
            raise ValueError("create_info section is required in configuration")

        root_path = Path(create_info_dict.get("root_path", ""))
        output_dir = Path(create_info_dict.get("output_dir", ""))
        model_list_raw = create_info_dict.get("model_list", [])
        model_list = [ModelConfig.from_cfg(m) for m in model_list_raw]

        return cls(root_path=root_path, output_dir=output_dir, model_list=model_list)


@dataclass(frozen=True)
class EnsembleInfosConfig:
    """Configuration for ensemble step."""

    config: Path

    @classmethod
    def from_cfg(cls, ensemble_dict: Dict[str, Any], base_dir: Path) -> "EnsembleInfosConfig":
        """Parse ensemble configuration from dictionary."""
        config = Path(ensemble_dict.get("config", ""))
        return cls(config=config)


@dataclass(frozen=True)
class CreatePseudoT4datasetConfig:
    """Configuration for pseudo_dataset step."""

    config: Path
    overwrite: bool

    @classmethod
    def from_cfg(cls, pseudo_dataset_dict: Dict[str, Any], base_dir: Path) -> "CreatePseudoT4datasetConfig":
        """Parse pseudo_dataset configuration from dictionary."""
        config = Path(pseudo_dataset_dict.get("config", ""))
        overwrite = pseudo_dataset_dict.get("overwrite", False)

        return cls(config=config, overwrite=overwrite)


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration."""

    logging: LoggingConfig
    create_info: CreateInfoConfig
    ensemble_infos: EnsembleInfosConfig
    create_pseudo_t4dataset: CreatePseudoT4datasetConfig

    @classmethod
    def from_cfg(cls, raw_config: Dict[str, Any], base_dir: Path) -> "PipelineConfig":
        """Parse pipeline configuration from dictionary."""
        logging_cfg = LoggingConfig.from_cfg(raw_config.get("logging", {}), base_dir)
        create_info_cfg = CreateInfoConfig.from_cfg(raw_config.get("create_info"), base_dir)
        ensemble_infos_cfg = EnsembleInfosConfig.from_cfg(raw_config.get("ensemble_infos", {}), base_dir)
        create_pseudo_t4dataset_cfg = CreatePseudoT4datasetConfig.from_cfg(raw_config.get("create_pseudo_t4dataset", {}), base_dir)

        return cls(
            logging=logging_cfg,
            create_info=create_info_cfg,
            ensemble_infos=ensemble_infos_cfg,
            create_pseudo_t4dataset=create_pseudo_t4dataset_cfg,
        )

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

        with config_path.open("r", encoding="utf-8") as fp:
            raw_config = yaml.safe_load(fp) or {}

        if not isinstance(raw_config, dict):
            raise TypeError("Top-level configuration must be a mapping")

        base_dir = config_path.parent

        return cls.from_cfg(raw_config, base_dir)


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
