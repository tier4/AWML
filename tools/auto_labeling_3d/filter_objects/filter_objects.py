import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

from mmengine.config import Config
from mmengine.registry import TASK_UTILS, init_default_scope

from tools.auto_labeling_3d.utils.logger import setup_logger


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


def filter_results(filter_cfg, predicted_result_info, predicted_result_info_name) -> Dict[str, Any]:
    """
    Args:
        filter_cfg (Dict[str, Any]): config for filter pipeline.
        predicted_result_info (Dict[str, Any]): Info dict that contains predicted result.
        predicted_result_info_name (str): Name of info dict.

    Returns:
        Dict[str, Any]: Filtered info dict
    """
    filter_model = TASK_UTILS.build(filter_cfg)
    return filter_model.filter(predicted_result_info, predicted_result_info_name)


def ensemble_results(ensemble_cfg: Dict[str, Any]):
    """
    Args:
        ensemble_cfg (Dict[str, Any]): config for ensemble.

    Returns:
        Dict[str, Any]: Ensembled info dict
    """
    ensemble_model = TASK_UTILS.build(ensemble_cfg.filter_pipelines)
    results: List[Dict[str, Any]] = []
    names = []
    for model_config in ensemble_model.models:
        names.append(model_config["name"])

        # load info file
        with open(model_config["info_path"], "rb") as f:
            info = pickle.load(f)

        # apply filters in pipelines
        for filter_cfg in model_config["filter_pipeline"]:
            info = filter_results(filter_cfg, info, model_config["info_path"])

        results.append(info)

    name = "+".join(names)
    output_info: Dict[str, Any] = ensemble_model.ensemble(results)
    return name, output_info


def main():
    init_default_scope("mmdet3d")

    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    logger: logging.Logger = setup_logger(args, name="filter_objects")

    # Apply filtering
    logger.info("Filtering objects...")
    if cfg.filter_pipelines.type == "EnsembleModel":
        name, output_info = ensemble_results(cfg)
    else:
        raise ValueError(f"Unknown filter type: {cfg.filter_pipelines.type}")

    # Get input info path and determine output path
    output_path = Path(args.work_dir) / f"pseudo_infos_{name}_filtered.pkl"

    # Save filtered results
    logger.info(f"Saving filtered results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(output_info, f)
    logger.info("Finish filtering")


if __name__ == "__main__":
    main()
