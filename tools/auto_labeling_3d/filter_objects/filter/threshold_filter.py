import logging
from collections import defaultdict
from typing import Any, Dict, List, Set

from mmengine.registry import TASK_UTILS

from .base_filter import BaseFilter


@TASK_UTILS.register_module()
class ThresholdFilter(BaseFilter):
    """Filter for removing low confidence predictions.

    Args:
        confidence_thresholds (Dict[str, float]): Confidence threshold for each class
        use_label (List[str]): List of labels to process
    """

    def __init__(self, confidence_thresholds: Dict[str, float], use_label: List[str], logger: logging.Logger):
        super().__init__(logger)
        self.settings = {
            "use_label": list(set(use_label)),
            "confidence_thresholds": confidence_thresholds,
        }

    def _should_filter_instance(self, pred_instance_3d: Dict[str, Any], category: str) -> bool:
        """Check if an instance should be filtered based on category and confidence.

        Args:
            pred_instance_3d (Dict[str, Any]): Prediction instance data
            category (str): Category name of the instance

        Returns:
            bool: True if instance should be filtered out, False otherwise
        """
        # Filter out if category is not in use_label
        if category not in self.settings["use_label"]:
            return True

        # Filter out if confidence is below threshold
        if pred_instance_3d["bbox_score_3d"] < self.settings["confidence_thresholds"][category]:
            return True

        return False

    def filter(self, predicted_result_info: Dict[str, Any], info_name: str) -> Dict[str, Any]:
        """Apply threshold filtering to the pseudo labels.

        Args:
            predicted_result_info (Dict[str, Any]): Info dict that contains predicted result.
            info_name (str): Name of each model used for generating info file.

        Returns:
            Dict[str, Any]: Filtered dataset info with filtering statistics
        """

        # Initialize counters
        total_instances = defaultdict(int)
        filtered_instances = defaultdict(int)

        # Filter predictions in each frame
        filtered_data_list = []
        for frame_info in predicted_result_info["data_list"]:
            filtered_pred_instances = []

            for pred_instance_3d in frame_info["pred_instances_3d"]:
                # Get category name
                category: str = predicted_result_info["metainfo"]["classes"][pred_instance_3d["bbox_label_3d"]]

                # Update total count
                total_instances[category] += 1

                # Apply filtering
                if not self._should_filter_instance(pred_instance_3d, category):
                    filtered_pred_instances.append(pred_instance_3d)
                else:
                    filtered_instances[category] += 1

            # Update frame with filtered predictions
            filtered_frame = frame_info.copy()
            filtered_frame["pred_instances_3d"] = filtered_pred_instances
            filtered_data_list.append(filtered_frame)

        # Create output
        filtered_predicted_result_info: Dict[str, Any] = predicted_result_info.copy()
        filtered_predicted_result_info["data_list"] = filtered_data_list
        self._report_filter_statistics(total_instances, filtered_instances, info_name)

        return filtered_predicted_result_info
