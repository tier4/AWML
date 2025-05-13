import logging
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from mmengine.registry import TASK_UTILS


@TASK_UTILS.register_module()
class EnsembleModel:
    """A class to ensemble the results of multiple detection models.

    Args:
        models (List[Dict]): A list of model configurations.
        ensemble_setting (Dict[str, Any]): Configuration for ensembling (e.g., weights, iou_threshold, skip_box_threshold).
    """

    def __init__(
        self,
        ensemble_setting: Dict[str, Any],
        logger: logging.Logger,
    ):
        self.settings = ensemble_setting
        self.logger = logger

    def ensemble(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble results from all model outputs.
        Args:
            results (List[Dict[str, Any]]): List of info dicts containing predicted results.

        Returns:
            Dict[str, Any]: Ensembled dataset info.
        """

        raise NotImplementedError("EnsembleModel is not implemented.")
