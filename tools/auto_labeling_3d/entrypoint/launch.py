import argparse
import logging
from pathlib import Path

from tools.auto_labeling_3d.entrypoint.parse_config import PipelineConfig, load_pipeline_config
from tools.auto_labeling_3d.script.download_checkpoints import download_file


def download_checkpoints(config: PipelineConfig, logger: logging.Logger) -> None:
    """
    Download checkpoints specified in the pipeline configuration.

    Args:
        config (PipelineConfig): The pipeline configuration containing model information.
        logger (logging.Logger): Logger for logging messages.
    """
    logger.info("Starting checkpoint download...")
    for model in config.create_info.model_list:
        url = model.checkpoint.model_zoo_url
        checkpoint_path = model.checkpoint.checkpoint_path
        if url and checkpoint_path:
            download_file(url, checkpoint_path, logger)
        else:
            logger.warning(f"Skipping model '{model.name}': missing url or checkpoint_path")
    logger.info("Checkpoint download completed.")


def trigger_auto_labeling_pipeline(config: PipelineConfig) -> None:
    # Execute the whole auto labeling pipeline.
    logger = logging.getLogger("auto_labeling_3d.entrypoint")
    
    # Step 1: Download checkpoints
    download_checkpoints(config, logger)


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
