import argparse
import logging
import pickle
from pathlib import Path

from mmengine.registry import init_default_scope

from tools.auto_labeling_3d.attach_tracking_id.attach_tracking_id import determine_scene_range, track_objects
from tools.auto_labeling_3d.change_directory_structure.change_directory_structure import process_dataset
from tools.auto_labeling_3d.create_info.create_info_data import create_info_data
from tools.auto_labeling_3d.create_pseudo_t4dataset.create_pseudo_t4dataset import create_pseudo_t4dataset
from tools.auto_labeling_3d.entrypoint.parse_config import (
    PipelineConfig,
    load_ensemble_config,
    load_model_config,
    load_t4dataset_config,
)
from tools.auto_labeling_3d.filter_objects.ensemble_infos import ensemble_infos
from tools.auto_labeling_3d.utils.download_checkpoint import download_checkpoint


def run_download_checkpoint(config: PipelineConfig, logger: logging.Logger) -> None:
    """
    Download checkpoints specified in the pipeline configuration.

    Args:
        config (PipelineConfig): The pipeline configuration containing model information.
        logger (logging.Logger): Logger for logging messages.
    """
    if config.create_info is None:
        raise ValueError("create_info configuration is required for run_download_checkpoint")

    logger.info("Starting checkpoint download...")
    for model in config.create_info.model_list:
        url = model.checkpoint.model_zoo_url
        checkpoint_path = model.checkpoint.checkpoint_path
        if url and checkpoint_path:
            download_checkpoint(url, checkpoint_path, logger)
        else:
            logger.warning(f"Skipping model '{model.name}': missing url or checkpoint_path")
    logger.info("Checkpoint download completed.")


def run_create_info_data(config: PipelineConfig, logger: logging.Logger) -> None:
    """
    Create info data for each model in the pipeline.

    Args:
        config (PipelineConfig): The pipeline configuration.
        logger (logging.Logger): Logger for logging messages.
    """
    if config.create_info is None:
        raise ValueError("create_info configuration is required for run_create_info_data")

    logger.info("Starting create_info_data step...")
    for model in config.create_info.model_list:
        logger.info(f"Processing model: {model.name}")

        # Load model config
        model_config = load_model_config(model, config.logging.work_dir)

        # Execute create_info_data
        create_info_data(
            non_annotated_dataset_path=config.root_path,
            model_config=model_config,
            model_checkpoint_path=str(model.checkpoint.checkpoint_path),
            model_name=model.name,
            out_dir=str(config.create_info.output_dir),
            logger=logger,
        )
        logger.info(f"Completed processing for model: {model.name}")

    logger.info("create_info_data step completed.")


def run_ensemble_infos(config: PipelineConfig, logger: logging.Logger) -> Path:
    """
    Ensemble infos from multiple models.

    Args:
        config (PipelineConfig): The pipeline configuration.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        Path: Path to the saved ensemble output file.

    Raises:
        ValueError: If ensemble_infos configuration is missing.
    """
    if config.ensemble_infos is None:
        raise ValueError("ensemble_infos configuration is required for run_ensemble_infos")

    logger.info("Starting ensemble step...")
    ensemble_cfg = load_ensemble_config(config.ensemble_infos.config)

    if ensemble_cfg.filter_pipelines.type != "Ensemble":
        raise ValueError(
            f"You cannot use {ensemble_cfg.filter_pipelines.type} type. Please use Ensemble type instead."
        )

    name, output_info = ensemble_infos(ensemble_cfg.filter_pipelines, logger)

    # Save ensembled results
    ensemble_output_path = config.logging.work_dir / f"pseudo_infos_{name}_filtered.pkl"
    logger.info(f"Saving filtered and ensembled results to {ensemble_output_path}")
    with open(ensemble_output_path, "wb") as f:
        pickle.dump(output_info, f)
    logger.info(f"Ensemble step completed. Results saved to {ensemble_output_path}")

    return ensemble_output_path


def run_attach_tracking_id(config: PipelineConfig, logger: logging.Logger, input_path: Path) -> Path:
    """
    Attach tracking IDs to pseudo labels.

    Args:
        config (PipelineConfig): The pipeline configuration.
        logger (logging.Logger): Logger for logging messages.
        input_path (Path): Path to the input file (ensemble output).

    Returns:
        Path: Path to the saved tracking output file.
    """
    logger.info("Starting tracking step...")
    tracking_output_path = config.logging.work_dir / "pseudo_infos_with_tracking.pkl"

    # Load dataset info
    with open(input_path, "rb") as f:
        dataset_info = pickle.load(f)

    # Determine scene boundaries and track objects
    scene_boundaries = determine_scene_range(dataset_info)
    for scene_boundary in scene_boundaries:
        dataset_info = track_objects(dataset_info, scene_boundary, logger)

    # Save tracked info
    with open(tracking_output_path, "wb") as f:
        pickle.dump(dataset_info, f)
    logger.info(f"Tracking step completed. Results saved to {tracking_output_path}")

    return tracking_output_path


def run_create_pseudo_t4dataset(config: PipelineConfig, logger: logging.Logger, input_path: Path) -> None:
    """
    Create pseudo T4dataset from tracked pseudo labels.

    Args:
        config (PipelineConfig): The pipeline configuration.
        logger (logging.Logger): Logger for logging messages.
        input_path (Path): Path to the input file (tracking output).

    Raises:
        ValueError: If create_pseudo_t4dataset or create_info configuration is missing.
    """
    if config.create_pseudo_t4dataset is None:
        raise ValueError("create_pseudo_t4dataset configuration is required for run_create_pseudo_t4dataset")

    logger.info("Starting create pseudo T4dataset step...")
    t4dataset_config = load_t4dataset_config(config.create_pseudo_t4dataset.config)

    create_pseudo_t4dataset(
        pseudo_labeled_info_path=input_path,
        non_annotated_dataset_path=config.root_path,
        t4dataset_config=t4dataset_config,
        overwrite=config.create_pseudo_t4dataset.overwrite,
        logger=logger,
    )
    logger.info("Create pseudo T4dataset step completed.")


def run_change_directory_structure(config: PipelineConfig, logger: logging.Logger) -> None:
    """
    Change directory structure for the pseudo T4dataset.

    Args:
        config (PipelineConfig): The pipeline configuration.
        logger (logging.Logger): Logger for logging messages.
    """
    if config.change_directory_structure is None:
        raise ValueError("change_directory_structure configuration is required for run_change_directory_structure")

    logger.info("Starting change_directory_structure step...")
    process_dataset(
        dataset_dir=config.root_path,
        logger=logger,
        annotated_to_non_annotated=False,
        version_dir_name=config.change_directory_structure.version_dir_name,
    )
    logger.info("Change directory structure step completed.")


def run_auto_labeling_pipeline(config: PipelineConfig) -> None:
    """Execute the whole auto labeling pipeline."""
    logger = logging.getLogger("auto_labeling_3d.entrypoint")

    # Ensure work directory exists
    config.logging.work_dir.mkdir(parents=True, exist_ok=True)

    if config.create_info:
        # Step 1: Download checkpoints
        run_download_checkpoint(config, logger)

        # Step 2: Create info data for each model
        run_create_info_data(config, logger)
    else:
        logger.warning(
            "Skipping download_checkpoint and create_info_data steps because create_info config is not contained in yaml."
        )

    if config.ensemble_infos and config.create_pseudo_t4dataset:
        # Step 3: Ensemble infos (only if configured)
        ensemble_output_path = run_ensemble_infos(config, logger)

        # Step 4: Attach tracking IDs
        tracking_output_path = run_attach_tracking_id(config, logger, ensemble_output_path)

        # Step 5: Create pseudo T4dataset
        run_create_pseudo_t4dataset(config, logger, tracking_output_path)
    else:
        logger.warning(
            "Skipping ensemble_infos, attach_tracking_id, and create_pseudo_t4dataset steps because ensemble_infos or create_pseudo_t4dataset config is not contained in yaml."
        )

    if config.change_directory_structure:
        # Step 6: Change directory structure
        run_change_directory_structure(config, logger)
    else:
        logger.warning(
            "Skipping change_directory_structure step because change_directory_structure config is not contained in yaml."
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
    # Initialize mmdet3d scope
    init_default_scope("mmdet3d")

    args = parse_args()
    config_path = Path(args.config).expanduser()
    pipeline_config = PipelineConfig.from_file(config_path)

    effective_level = args.log_level or pipeline_config.logging.level
    logging.basicConfig(level=getattr(logging, effective_level.upper(), logging.INFO))
    logger = logging.getLogger("auto_labeling_3d.entrypoint")

    run_auto_labeling_pipeline(pipeline_config)


if __name__ == "__main__":
    main()
