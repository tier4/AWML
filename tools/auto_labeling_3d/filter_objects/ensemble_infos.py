import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

from mmengine.config import Config
from mmengine.registry import TASK_UTILS, init_default_scope

from tools.auto_labeling_3d.filter_objects.filter_objects import filter_result
from tools.auto_labeling_3d.utils.logger import setup_logger


def apply_ensemble(
    ensemble_cfg: Dict[str, Any], predicted_result_infos: List[Dict[str, Any]], logger: logging.Logger
) -> Dict[str, Any]:
    """
    Args:
        ensemble_cfg (Dict[str, Any]): config for ensemble model.
        predicted_result_infos (List[Dict[str, Any]]): List of info dict that contains predicted result.
        logger (logging.Logger): Logger instance for output messages.

    Returns:
        Dict[str, Any]: Ensembled info dict
    """
    ensemble_cfg["logger"] = logger
    ensemble_model: EnsembleModel = TASK_UTILS.build(ensemble_cfg)
    return ensemble_model.ensemble(predicted_result_infos)


def ensemble_infos(filter_pipelines: Dict[str, Any], logger: logging.Logger) -> tuple[str, Dict[str, Any]]:
    """
    Args:
        filter_pipelines (Dict[str, Any]): config for pipelines.
        logger (logging.Logger): Logger instance for output messages.

    Returns:
        str: Name of models ensembled (e.g. "centerpoint+bevfusion")
        Dict[str, Any]: Ensembled info dict
    """
    names: List[str] = []
    predicted_results: List[Dict[str, Any]] = []
    for filter_input in filter_pipelines.inputs:
        name, info = filter_result(filter_input, logger)

        names.append(name)
        predicted_results.append(info)

    name: str = "+".join(names)
    output_info: Dict[str, Any] = apply_ensemble(filter_pipelines.config, predicted_results, logger)
    return name, output_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter objects which do not use for pseudo T4dataset from pseudo labeled info file"
    )

    parser.add_argument("--config", type=str, required=True, help="Path to config file of the filtering")
    parser.add_argument(
        "--work-dir", required=True, help="the directory to save the file containing evaluation metrics"
    )
    parser.add_argument(
        "--log-level",
        help="Set log level",
        default="INFO",
        choices=list(logging._nameToLevel.keys()),
    )
    return parser.parse_args()


def main():
    # setup
    init_default_scope("mmdet3d")
    args = parse_args()
    logger: logging.Logger = setup_logger(args, name="filter_objects")

    # Load config
    cfg = Config.fromfile(args.config)

    # Ensemble objects
    logger.info("Ensembling infos...")
    if cfg.filter_pipelines.type == "Ensemble":
        name, output_info = ensemble_infos(cfg.filter_pipelines, logger)
    else:
        raise ValueError(f"You cannot use {cfg.filter_pipelines.type} type. Please use Ensemble type instead.")

    # Save filtered results
    output_path = Path(args.work_dir) / f"pseudo_infos_{name}_filtered.pkl"
    logger.info(f"Saving filtered and ensembled results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(output_info, f)
    logger.info("Finish ensembling")


if __name__ == "__main__":
    main()
