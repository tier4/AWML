import argparse
import logging
import pickle
from pathlib import Path

from mmengine.registry import init_default_scope

from tools.auto_labeling_3d.attach_tracking_id.attach_tracking_id import determine_scene_range, track_objects
from tools.auto_labeling_3d.create_info.create_info_data import create_info_data
from tools.auto_labeling_3d.create_pseudo_t4dataset.create_pseudo_t4dataset import create_pseudo_t4dataset
from tools.auto_labeling_3d.entrypoint.parse_config import (
    PipelineConfig,
    load_ensemble_config,
    load_model_config,
    load_pipeline_config,
    load_t4dataset_config,
)
from tools.auto_labeling_3d.filter_objects.ensemble_infos import ensemble_infos
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

    # Step 2: Create info data for each model
    logger.info("Starting create_info_data step...")
    for model in config.create_info.model_list:
        logger.info(f"Processing model: {model.name}")

        # Load model config
        model_config = load_model_config(model, config.logging.work_dir)

        # Execute create_info_data
        create_info_data(
            non_annotated_dataset_path=config.create_info.root_path,
            model_config=model_config,
            model_checkpoint_path=str(model.checkpoint.checkpoint_path),
            model_name=model.name,
            out_dir=str(config.create_info.output_dir),
            logger=logger,
        )
        logger.info(f"Completed processing for model: {model.name}")

    logger.info("create_info_data step completed.")

    # Step 3: Ensemble infos
    logger.info("Starting ensemble step...")
    ensemble_cfg = load_ensemble_config(config.ensemble_infos.config)

    if ensemble_cfg.filter_pipelines.type == "Ensemble":
        name, output_info = ensemble_infos(ensemble_cfg.filter_pipelines, logger)

        # Save ensembled results
        ensemble_output_path = config.logging.work_dir / f"pseudo_infos_{name}_filtered.pkl"
        logger.info(f"Saving filtered and ensembled results to {ensemble_output_path}")
        with open(ensemble_output_path, "wb") as f:
            pickle.dump(output_info, f)
        logger.info(f"Ensemble step completed. Results saved to {ensemble_output_path}")
    else:
        raise ValueError(
            f"You cannot use {ensemble_cfg.filter_pipelines.type} type. Please use Ensemble type instead."
        )

    # Step 4: Attach tracking IDs
    logger.info("Starting tracking step...")
    tracking_input_path = ensemble_output_path  # Use output from ensemble step
    tracking_output_path = config.logging.work_dir / "pseudo_infos_with_tracking.pkl"

    # Load dataset info
    with open(tracking_input_path, "rb") as f:
        dataset_info = pickle.load(f)

    # Determine scene boundaries and track objects
    scene_boundaries = determine_scene_range(dataset_info)
    for scene_boundary in scene_boundaries:
        dataset_info = track_objects(dataset_info, scene_boundary, logger)

    # Save tracked info
    with open(tracking_output_path, "wb") as f:
        pickle.dump(dataset_info, f)
    logger.info(f"Tracking step completed. Results saved to {tracking_output_path}")

    # Step 5: Create pseudo T4dataset
    logger.info("Starting create pseudo T4dataset step...")
    t4dataset_config = load_t4dataset_config(config.create_pseudo_t4dataset.config)

    create_pseudo_t4dataset(
        pseudo_labeled_info_path=tracking_output_path,
        non_annotated_dataset_path=config.create_info.root_path,
        t4dataset_config=t4dataset_config,
        overwrite=config.create_pseudo_t4dataset.overwrite,
        logger=logger,
    )
    logger.info("Create pseudo T4dataset step completed.")


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
    pipeline_config = load_pipeline_config(config_path)

    effective_level = args.log_level or pipeline_config.logging.level
    logging.basicConfig(level=getattr(logging, effective_level.upper(), logging.INFO))
    logger = logging.getLogger("auto_labeling_3d.entrypoint")

    trigger_auto_labeling_pipeline(pipeline_config)


if __name__ == "__main__":
    main()
