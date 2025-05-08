import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

from mmengine.config import Config
from mmengine.registry import TASK_UTILS, init_default_scope

from tools.auto_labeling_3d.utils.logger import setup_logger
from tools.auto_labeling_3d.utils.type import AWML3DInfo

def apply_filter(filter_cfg: Dict[str, Any], predicted_result_info: AWML3DInfo, predicted_result_info_name: str, logger: logging.Logger) -> AWML3DInfo:
    """
    Args:
        filter_cfg (Dict[str, Any]): config for filter pipeline.
        predicted_result_info (AWML3DInfo): AWML3DInfo dict that contains predicted result.
        predicted_result_info_name (str): Name of AWML3DInfo dict.
        logger (logging.Logger): Logger instance for output messages.

    Returns:
       AWML3DInfo: Filtered info dict
    """
    filter_cfg['logger'] = logger
    filter_model = TASK_UTILS.build(filter_cfg)
    return filter_model.filter(predicted_result_info, predicted_result_info_name)

def apply_ensemble(ensemble_cfg: Dict[str, Any], predicted_result_infos: List[AWML3DInfo], logger: logging.Logger) -> AWML3DInfo:
    """
    Args:
        ensemble_cfg (Dict[str, Any]): config for ensemble model.
        predicted_result_infos (List[AWML3DInfo]): List of AWML3DInfo dict that contains predicted result.
        logger (logging.Logger): Logger instance for output messages.

    Returns:
        AWML3DInfo: Ensembled info dict
    """
    ensemble_cfg['logger'] = logger
    ensemble_model: EnsembleModel = TASK_UTILS.build(ensemble_cfg)
    return ensemble_model.ensemble(predicted_result_infos)

def filter_result(filter_input: Dict[str, Any], logger: logging.Logger) -> tuple[str, AWML3DInfo]:
    """
    Args:
        filter_input (Dict[str, Any]): config of input for filter.
        logger (logging.Logger): Logger instance for output messages.

    Returns:
        str: Name of the model used for input
        AWML3DInfo: Filtered info dict
    """
    # load info file
    with open(filter_input["info_path"], "rb") as f:
        info: AWML3DInfo = pickle.load(f)

    # apply filters in pipelines
    for filter_cfg in filter_input["filter_pipeline"]:
        info: AWML3DInfo = apply_filter(filter_cfg, info, filter_input["info_path"], logger)

    name: str = filter_input["name"]
    output_info: AWML3DInfo = info
    return name, output_info

def ensemble_results(filter_pipelines: Dict[str, Any], logger: logging.Logger) -> tuple[str, AWML3DInfo]:
    """
    Args:
        filter_pipelines (Dict[str, Any]): config for pipelines.
        logger (logging.Logger): Logger instance for output messages.

    Returns:
        str: Name of models ensembled (e.g. "centerpoint+bevfusion")
        AWML3DInfo: Ensembled info dict
    """
    names: List[str] = []
    predicted_results: List[AWML3DInfo] = []
    for filter_input in filter_pipelines.inputs:
        name, info = filter_result(filter_input, logger)

        names.append(name)
        predicted_results.append(info)

    name: str = "+".join(names)
    output_info: AWML3DInfo = apply_ensemble(filter_pipelines.config, predicted_results, logger)
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
    match cfg.filter_pipelines.type:
        case "Ensemble":
            name, output_info = ensemble_results(cfg.filter_pipelines, logger)
        case "Filter":
            name, output_info = filter_result(cfg.filter_pipelines.input, logger)
        case _:
            raise ValueError(f"Unknown filter type: {cfg.filter_pipelines.type}")

    # Save filtered results
    output_path = Path(args.work_dir) / f"pseudo_infos_{name}_filtered.pkl"
    logger.info(f"Saving filtered results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(output_info, f)
    logger.info("Finish filtering")

if __name__ == "__main__":
    main()
