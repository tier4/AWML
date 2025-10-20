import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def trigger_auto_labeling_pipeline(config: "PipelineConfig") -> None:
    # Execute the whole auto labeling pipeline.
    pass


def load_pipeline_config(config_path: Path) -> PipelineConfig:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto Labeling 3D pipeline launcher",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the pipeline YAML configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override logging level (e.g., DEBUG, INFO, WARNING)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    pipeline_config = load_pipeline_config(config_path)

    effective_level = args.log_level or pipeline_config.logging.level
    logging.basicConfig(level=getattr(logging, effective_level.upper(), logging.INFO))
    logger = logging.getLogger("auto_labeling_3d.entrypoint")

    trigger_auto_labeling_pipeline(pipeline_config)


if __name__ == "__main__":
    main()
