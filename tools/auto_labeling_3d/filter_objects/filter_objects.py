import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

from mmengine.config import Config
from mmengine.registry import TASK_UTILS, init_default_scope

from tools.auto_labeling_3d.utils.logger import setup_logger


def apply_filter(
    filter_cfg: Dict[str, Any],
    predicted_result_info: Dict[str, Any],
    predicted_result_info_name: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Args:
        filter_cfg (Dict[str, Any]): config for filter pipeline.
        predicted_result_info (Dict[str, Any]): info dict that contains predicted result.
        predicted_result_info_name (str): Name of info dict.
        logger (logging.Logger): Logger instance for output messages.

    Returns:
       Dict[str, Any]: Filtered info dict
    """
    filter_cfg["logger"] = logger
    filter_model = TASK_UTILS.build(filter_cfg)
    return filter_model.filter(predicted_result_info, predicted_result_info_name)


def filter_result(filter_input: Dict[str, Any], logger: logging.Logger) -> tuple[str, Dict[str, Any]]:
    """
    Args:
        filter_input (Dict[str, Any]): config of input for filter.
        logger (logging.Logger): Logger instance for output messages.

    Returns:
        str: Name of the model used for input
        Dict[str, Any]: Filtered info dict
    """
    # load info file
    with open(filter_input["info_path"], "rb") as f:
        info: Dict[str, Any] = pickle.load(f)

    # apply filters in pipelines
    for filter_cfg in filter_input["filter_pipeline"]:
        info: Dict[str, Any] = apply_filter(filter_cfg, info, filter_input["info_path"], logger)

    name: str = filter_input["name"]
    output_info: Dict[str, Any] = info
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

    # Filter objects
    logger.info("Filtering objects...")
    if cfg.filter_pipelines.type == "Filter":
        name, output_info = filter_result(cfg.filter_pipelines.input, logger)
    else:
        raise ValueError(f"You cannot use {cfg.filter_pipelines.type} type. Please use Filter type instead.")

    # Save filtered results
    output_path = Path(args.work_dir) / f"pseudo_infos_{name}_filtered.pkl"
    logger.info(f"Saving filtered results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(output_info, f)
    logger.info("Finish filtering")


if __name__ == "__main__":
    main()
