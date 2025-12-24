"""
2D Detection Metrics Interface using autoware_perception_evaluation.

This module provides an interface to compute 2D detection metrics (mAP)
using autoware_perception_evaluation in 2D mode, ensuring consistent metrics
between training evaluation and deployment evaluation.

For 2D detection, the interface uses:
- IoU 2D thresholds for matching (e.g., 0.5, 0.75)
- Only AP is computed (no APH since there's no heading in 2D)

Usage:
    config = Detection2DMetricsConfig(
        class_names=["car", "truck", "bus", "bicycle", "pedestrian", "motorcycle", "trailer", "unknown"],
        frame_id="camera",
    )
    interface = Detection2DMetricsInterface(config)

    # Add frames
    for pred, gt in zip(predictions_list, ground_truths_list):
        interface.add_frame(
            predictions=pred,  # List[Dict] with bbox (x1,y1,x2,y2), label, score
            ground_truths=gt,  # List[Dict] with bbox (x1,y1,x2,y2), label
        )

    # Compute metrics
    metrics = interface.compute_metrics()
    # Returns: {"mAP_iou_2d_0.5": 0.7, "mAP_iou_2d_0.75": 0.65, ...}
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel, Label
from perception_eval.common.object2d import DynamicObject2D
from perception_eval.common.schema import FrameID
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)
from perception_eval.manager import PerceptionEvaluationManager

from deployment.core.metrics.base_metrics_interface import BaseMetricsConfig, BaseMetricsInterface, DetectionSummary

logger = logging.getLogger(__name__)


# Valid 2D frame IDs for camera-based detection
VALID_2D_FRAME_IDS = [
    "cam_front",
    "cam_front_right",
    "cam_front_left",
    "cam_front_lower",
    "cam_back",
    "cam_back_left",
    "cam_back_right",
    "cam_traffic_light_near",
    "cam_traffic_light_far",
    "cam_traffic_light",
]


@dataclass(frozen=True)
class Detection2DMetricsConfig(BaseMetricsConfig):
    """Configuration for 2D detection metrics.

    Attributes:
        class_names: List of class names for evaluation.
        frame_id: Frame ID for evaluation. Valid values for 2D:
            "cam_front", "cam_front_right", "cam_front_left", "cam_front_lower",
            "cam_back", "cam_back_left", "cam_back_right",
            "cam_traffic_light_near", "cam_traffic_light_far", "cam_traffic_light"
        iou_thresholds: List of IoU thresholds for evaluation.
        evaluation_config_dict: Configuration dict for perception evaluation.
        critical_object_filter_config: Config for filtering critical objects.
        frame_pass_fail_config: Config for pass/fail criteria.
    """

    # Override default frame_id for 2D detection (camera frame instead of base_link)
    frame_id: str = "cam_front"
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75])
    evaluation_config_dict: Optional[Dict[str, Any]] = None
    critical_object_filter_config: Optional[Dict[str, Any]] = None
    frame_pass_fail_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Validate frame_id for 2D detection
        if self.frame_id not in VALID_2D_FRAME_IDS:
            raise ValueError(
                f"Invalid frame_id '{self.frame_id}' for 2D detection. Valid options: {VALID_2D_FRAME_IDS}"
            )

        # Set default evaluation config if not provided
        if self.evaluation_config_dict is None:
            default_eval_config = {
                "evaluation_task": "detection2d",
                "target_labels": self.class_names,
                "iou_2d_thresholds": self.iou_thresholds,
                "center_distance_bev_thresholds": None,
                "plane_distance_thresholds": None,
                "iou_3d_thresholds": None,
                "label_prefix": "autoware",
            }
            object.__setattr__(self, "evaluation_config_dict", default_eval_config)

        # Set default critical object filter config if not provided
        if self.critical_object_filter_config is None:
            default_filter_config = {
                "target_labels": self.class_names,
                "ignore_attributes": None,
            }
            object.__setattr__(self, "critical_object_filter_config", default_filter_config)

        # Set default frame pass fail config if not provided
        if self.frame_pass_fail_config is None:
            num_classes = len(self.class_names)
            default_pass_fail_config = {
                "target_labels": self.class_names,
                "matching_threshold_list": [0.5] * num_classes,
                "confidence_threshold_list": None,
            }
            object.__setattr__(self, "frame_pass_fail_config", default_pass_fail_config)


class Detection2DMetricsInterface(BaseMetricsInterface):
    """
    Interface for computing 2D detection metrics using autoware_perception_evaluation.

    This interface provides a simplified interface for the deployment framework to
    compute mAP for 2D object detection tasks (YOLOX, etc.).

    Unlike 3D detection, 2D detection:
    - Uses IoU 2D for matching (based on bounding box overlap)
    - Does not compute APH (no heading information in 2D)
    - Works with image-space bounding boxes [x1, y1, x2, y2]

    Example usage:
        config = Detection2DMetricsConfig(
            class_names=["car", "truck", "bus", "bicycle", "pedestrian"],
            iou_thresholds=[0.5, 0.75],
        )
        interface = Detection2DMetricsInterface(config)

        # Add frames
        for pred, gt in zip(predictions_list, ground_truths_list):
            interface.add_frame(
                predictions=pred,  # List[Dict] with bbox, label, score
                ground_truths=gt,  # List[Dict] with bbox, label
            )

        # Compute metrics
        metrics = interface.compute_metrics()
    """

    _UNKNOWN = "unknown"

    def __init__(
        self,
        config: Detection2DMetricsConfig,
        data_root: str = "data/t4dataset/",
        result_root_directory: str = "/tmp/perception_eval_2d/",
    ):
        """
        Initialize the 2D detection metrics interface.

        Args:
            config: Configuration for 2D detection metrics.
            data_root: Root directory of the dataset.
            result_root_directory: Directory for saving evaluation results.
        """
        super().__init__(config)
        self.config: Detection2DMetricsConfig = config
        self.data_root = data_root
        self.result_root_directory = result_root_directory

        # Create perception evaluation config
        self.perception_eval_config = PerceptionEvaluationConfig(
            dataset_paths=data_root,
            frame_id=config.frame_id,
            result_root_directory=result_root_directory,
            evaluation_config_dict=config.evaluation_config_dict,
            load_raw_data=False,
        )

        # Create critical object filter config
        self.critical_object_filter_config = CriticalObjectFilterConfig(
            evaluator_config=self.perception_eval_config,
            **config.critical_object_filter_config,
        )

        # Create frame pass fail config
        self.frame_pass_fail_config = PerceptionPassFailConfig(
            evaluator_config=self.perception_eval_config,
            **config.frame_pass_fail_config,
        )

        # Initialize evaluation manager
        self.evaluator: Optional[PerceptionEvaluationManager] = None

    def reset(self) -> None:
        """Reset the interface for a new evaluation session."""
        self.evaluator = PerceptionEvaluationManager(
            evaluation_config=self.perception_eval_config,
            load_ground_truth=False,
            metric_output_dir=None,
        )
        self._frame_count = 0

    def _convert_index_to_label(self, label_index: int) -> Label:
        """Convert a label index to a Label object.

        Args:
            label_index: Index of the label in class_names.

        Returns:
            Label object with AutowareLabel.
        """
        if 0 <= label_index < len(self.class_names):
            class_name = self.class_names[label_index]
        else:
            class_name = self._UNKNOWN

        autoware_label = AutowareLabel.__members__.get(class_name.upper(), AutowareLabel.UNKNOWN)
        return Label(label=autoware_label, name=class_name)

    def _predictions_to_dynamic_objects_2d(
        self,
        predictions: List[Dict[str, Any]],
        unix_time: int,
    ) -> List[DynamicObject2D]:
        """Convert prediction dicts to DynamicObject2D instances.

        Args:
            predictions: List of prediction dicts with keys:
                - bbox: [x1, y1, x2, y2] (image coordinates)
                - label: int (class index)
                - score: float (confidence score)
            unix_time: Unix timestamp in microseconds.

        Returns:
            List of DynamicObject2D instances.
        """
        estimated_objects = []
        frame_id = FrameID.from_value(self.frame_id)

        for pred in predictions:
            bbox = pred.get("bbox", [])
            if len(bbox) < 4:
                continue

            # Extract bbox components [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Convert [x1, y1, x2, y2] to [xmin, ymin, width, height] format
            # as required by DynamicObject2D.roi
            xmin = int(x1)
            ymin = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)

            # Get label
            label_idx = pred.get("label", 0)
            semantic_label = self._convert_index_to_label(int(label_idx))

            # Get score
            score = float(pred.get("score", 0.0))

            # Create DynamicObject2D
            # roi format: (xmin, ymin, width, height)
            dynamic_obj = DynamicObject2D(
                unix_time=unix_time,
                frame_id=frame_id,
                semantic_score=score,
                semantic_label=semantic_label,
                roi=(xmin, ymin, width, height),
                uuid=None,
            )
            estimated_objects.append(dynamic_obj)

        return estimated_objects

    def _ground_truths_to_dynamic_objects_2d(
        self,
        ground_truths: List[Dict[str, Any]],
        unix_time: int,
    ) -> List[DynamicObject2D]:
        """Convert ground truth dicts to DynamicObject2D instances.

        Args:
            ground_truths: List of ground truth dicts with keys:
                - bbox: [x1, y1, x2, y2] (image coordinates)
                - label: int (class index)
            unix_time: Unix timestamp in microseconds.

        Returns:
            List of DynamicObject2D instances.
        """
        gt_objects = []
        frame_id = FrameID.from_value(self.frame_id)

        for gt in ground_truths:
            bbox = gt.get("bbox", [])
            if len(bbox) < 4:
                continue

            # Extract bbox components [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Convert [x1, y1, x2, y2] to [xmin, ymin, width, height] format
            # as required by DynamicObject2D.roi
            xmin = int(x1)
            ymin = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)

            # Get label
            label_idx = gt.get("label", 0)
            semantic_label = self._convert_index_to_label(int(label_idx))

            # Create DynamicObject2D (GT always has score 1.0)
            # roi format: (xmin, ymin, width, height)
            dynamic_obj = DynamicObject2D(
                unix_time=unix_time,
                frame_id=frame_id,
                semantic_score=1.0,
                semantic_label=semantic_label,
                roi=(xmin, ymin, width, height),
                uuid=None,
            )
            gt_objects.append(dynamic_obj)

        return gt_objects

    def add_frame(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        frame_name: Optional[str] = None,
    ) -> None:
        """Add a frame of predictions and ground truths for evaluation.

        Args:
            predictions: List of prediction dicts with keys:
                - bbox: [x1, y1, x2, y2] (image coordinates)
                - label: int (class index)
                - score: float (confidence score)
            ground_truths: List of ground truth dicts with keys:
                - bbox: [x1, y1, x2, y2] (image coordinates)
                - label: int (class index)
            frame_name: Optional name for the frame.
        """
        if self.evaluator is None:
            self.reset()

        # Unix time in microseconds (int)
        unix_time = int(time.time() * 1e6)
        if frame_name is None:
            frame_name = str(self._frame_count)

        # Convert predictions to DynamicObject2D
        estimated_objects = self._predictions_to_dynamic_objects_2d(predictions, unix_time)

        # Convert ground truths to DynamicObject2D list
        gt_objects = self._ground_truths_to_dynamic_objects_2d(ground_truths, unix_time)

        # Create FrameGroundTruth for 2D
        frame_ground_truth = FrameGroundTruth(
            unix_time=unix_time,
            frame_name=frame_name,
            objects=gt_objects,
            transforms=None,
            raw_data=None,
        )

        # Add frame result to evaluator
        try:
            self.evaluator.add_frame_result(
                unix_time=unix_time,
                ground_truth_now_frame=frame_ground_truth,
                estimated_objects=estimated_objects,
                critical_object_filter_config=self.critical_object_filter_config,
                frame_pass_fail_config=self.frame_pass_fail_config,
            )
            self._frame_count += 1
        except Exception as e:
            logger.warning(f"Failed to add frame {frame_name}: {e}")

    def compute_metrics(self) -> Dict[str, float]:
        """Compute metrics from all added frames.

        Returns:
            Dictionary of metrics with keys like:
                - mAP_iou_2d_0.5
                - mAP_iou_2d_0.75
                - car_AP_iou_2d_0.5
                - etc.
        """
        if self.evaluator is None or self._frame_count == 0:
            logger.warning("No frames to evaluate")
            return {}

        try:
            # Get scene result (aggregated metrics)
            metrics_score: MetricsScore = self.evaluator.get_scene_result()

            # Process metrics into a flat dictionary
            return self._process_metrics_score(metrics_score)

        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _process_metrics_score(self, metrics_score: MetricsScore) -> Dict[str, float]:
        """Process MetricsScore into a flat dictionary.

        Args:
            metrics_score: MetricsScore instance from evaluator.

        Returns:
            Flat dictionary of metrics.
        """
        metric_dict = {}

        for map_instance in metrics_score.mean_ap_values:
            matching_mode = map_instance.matching_mode.value.lower().replace(" ", "_")

            # Process individual AP values
            for label, aps in map_instance.label_to_aps.items():
                label_name = label.value

                for ap in aps:
                    threshold = ap.matching_threshold
                    ap_value = ap.ap

                    # Create the metric key
                    key = f"{label_name}_AP_{matching_mode}_{threshold}"
                    metric_dict[key] = ap_value

            # Add mAP value (no mAPH for 2D detection)
            map_key = f"mAP_{matching_mode}"
            metric_dict[map_key] = map_instance.map

        return metric_dict

    def get_summary(self) -> DetectionSummary:
        """Get a summary of the evaluation including mAP and per-class metrics."""
        metrics = self.compute_metrics()

        # Extract primary metrics (first mAP value found)
        primary_map = None
        per_class_ap = {}

        for key, value in metrics.items():
            if key.startswith("mAP_") and primary_map is None:
                primary_map = value
            elif "_AP_" in key and not key.startswith("mAP"):
                # Extract class name from key
                parts = key.split("_AP_")
                if len(parts) == 2:
                    class_name = parts[0]
                    if class_name not in per_class_ap:
                        per_class_ap[class_name] = value

        return DetectionSummary(
            mAP=primary_map or 0.0,
            per_class_ap=per_class_ap,
            num_frames=self._frame_count,
            detailed_metrics=metrics,
        )
