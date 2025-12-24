"""
3D Detection Metrics Interface using autoware_perception_evaluation.

This module provides an interface to compute 3D detection metrics (mAP, mAPH)
using autoware_perception_evaluation, ensuring consistent metrics between
training evaluation (T4MetricV2) and deployment evaluation.

Usage:
    config = Detection3DMetricsConfig(
        class_names=["car", "truck", "bus", "bicycle", "pedestrian"],
        frame_id="base_link",
    )
    interface = Detection3DMetricsInterface(config)

    # Add frames
    for pred, gt in zip(predictions_list, ground_truths_list):
        interface.add_frame(
            predictions=pred,  # List[Dict] with bbox_3d, label, score
            ground_truths=gt,  # List[Dict] with bbox_3d, label
        )

    # Compute metrics
    metrics = interface.compute_metrics()
    # Returns: {"mAP_center_distance_bev_0.5": 0.7, ...}
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel, Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.shape import Shape, ShapeType
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)
from perception_eval.manager import PerceptionEvaluationManager
from pyquaternion import Quaternion

from deployment.core.metrics.base_metrics_interface import BaseMetricsConfig, BaseMetricsInterface, DetectionSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection3DMetricsConfig(BaseMetricsConfig):
    """Configuration for 3D detection metrics.

    Attributes:
        class_names: List of class names for evaluation.
        frame_id: Frame ID for evaluation (e.g., "base_link").
        evaluation_config_dict: Configuration dict for perception evaluation.
            Example:
                {
                    "evaluation_task": "detection",
                    "target_labels": ["car", "truck", "bus", "bicycle", "pedestrian"],
                    "center_distance_bev_thresholds": [0.5, 1.0, 2.0, 4.0],
                    "plane_distance_thresholds": [2.0, 4.0],
                    "iou_2d_thresholds": None,
                    "iou_3d_thresholds": None,
                    "label_prefix": "autoware",
                    "max_distance": 121.0,
                    "min_distance": -121.0,
                    "min_point_numbers": 0,
                }
        critical_object_filter_config: Config for filtering critical objects.
            Example:
                {
                    "target_labels": ["car", "truck", "bus", "bicycle", "pedestrian"],
                    "ignore_attributes": None,
                    "max_distance_list": [121.0, 121.0, 121.0, 121.0, 121.0],
                    "min_distance_list": [-121.0, -121.0, -121.0, -121.0, -121.0],
                }
        frame_pass_fail_config: Config for pass/fail criteria.
            Example:
                {
                    "target_labels": ["car", "truck", "bus", "bicycle", "pedestrian"],
                    "matching_threshold_list": [2.0, 2.0, 2.0, 2.0, 2.0],
                    "confidence_threshold_list": None,
                }
    """

    evaluation_config_dict: Optional[Dict[str, Any]] = None
    critical_object_filter_config: Optional[Dict[str, Any]] = None
    frame_pass_fail_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Set default evaluation config if not provided
        if self.evaluation_config_dict is None:
            default_eval_config = {
                "evaluation_task": "detection",
                "target_labels": self.class_names,
                "center_distance_bev_thresholds": [0.5, 1.0, 2.0, 4.0],
                "plane_distance_thresholds": [2.0, 4.0],
                "iou_2d_thresholds": None,
                "iou_3d_thresholds": None,
                "label_prefix": "autoware",
                "max_distance": 121.0,
                "min_distance": -121.0,
                "min_point_numbers": 0,
            }
            object.__setattr__(self, "evaluation_config_dict", default_eval_config)

        # Set default critical object filter config if not provided
        if self.critical_object_filter_config is None:
            num_classes = len(self.class_names)
            default_filter_config = {
                "target_labels": self.class_names,
                "ignore_attributes": None,
                "max_distance_list": [121.0] * num_classes,
                "min_distance_list": [-121.0] * num_classes,
            }
            object.__setattr__(self, "critical_object_filter_config", default_filter_config)

        # Set default frame pass fail config if not provided
        if self.frame_pass_fail_config is None:
            num_classes = len(self.class_names)
            default_pass_fail_config = {
                "target_labels": self.class_names,
                "matching_threshold_list": [2.0] * num_classes,
                "confidence_threshold_list": None,
            }
            object.__setattr__(self, "frame_pass_fail_config", default_pass_fail_config)


class Detection3DMetricsInterface(BaseMetricsInterface):
    """
    Interface for computing 3D detection metrics using autoware_perception_evaluation.

    This interface provides a simplified interface for the deployment framework to
    compute mAP, mAPH, and other detection metrics that are consistent with
    the T4MetricV2 used during training.

    Example usage:
        config = Detection3DMetricsConfig(
            class_names=["car", "truck", "bus", "bicycle", "pedestrian"],
            frame_id="base_link",
        )
        interface = Detection3DMetricsInterface(config)

        # Add frames
        for pred, gt in zip(predictions_list, ground_truths_list):
            interface.add_frame(
                predictions=pred,  # List[Dict] with bbox_3d, label, score
                ground_truths=gt,  # List[Dict] with bbox_3d, label
            )

        # Compute metrics
        metrics = interface.compute_metrics()
        # Returns: {"mAP_center_distance_bev_0.5": 0.7, ...}
    """

    _UNKNOWN = "unknown"

    def __init__(
        self,
        config: Detection3DMetricsConfig,
        data_root: str = "data/t4dataset/",
        result_root_directory: str = "/tmp/perception_eval/",
    ):
        """
        Initialize the 3D detection metrics interface.

        Args:
            config: Configuration for 3D detection metrics.
            data_root: Root directory of the dataset.
            result_root_directory: Directory for saving evaluation results.
        """
        super().__init__(config)
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

        # Initialize evaluation manager (will be created on first use or reset)
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

    def _predictions_to_dynamic_objects(
        self,
        predictions: List[Dict[str, Any]],
        unix_time: float,
    ) -> List[DynamicObject]:
        """Convert prediction dicts to DynamicObject instances.

        Args:
            predictions: List of prediction dicts with keys:
                - bbox_3d: [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, vx, vy]
                  (Same format as mmdet3d LiDARInstance3DBoxes)
                - label: int (class index)
                - score: float (confidence score)
            unix_time: Unix timestamp for the frame.

        Returns:
            List of DynamicObject instances.
        """
        estimated_objects = []
        for pred in predictions:
            bbox = pred.get("bbox_3d", [])
            if len(bbox) < 7:
                continue

            # Extract bbox components
            # mmdet3d LiDARInstance3DBoxes format: [x, y, z, l, w, h, yaw, vx, vy]
            # where l=length, w=width, h=height
            x, y, z = bbox[0], bbox[1], bbox[2]
            l, w, h = bbox[3], bbox[4], bbox[5]
            yaw = bbox[6]

            # Velocity (optional)
            vx = bbox[7] if len(bbox) > 7 else 0.0
            vy = bbox[8] if len(bbox) > 8 else 0.0

            # Create quaternion from yaw
            orientation = Quaternion(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))

            # Get label
            label_idx = pred.get("label", 0)
            semantic_label = self._convert_index_to_label(int(label_idx))

            # Get score
            score = float(pred.get("score", 0.0))

            # Shape size follows autoware_perception_evaluation convention: (length, width, height)
            dynamic_obj = DynamicObject(
                unix_time=unix_time,
                frame_id=self.frame_id,
                position=(x, y, z),
                orientation=orientation,
                shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(l, w, h)),
                velocity=(vx, vy, 0.0),
                semantic_score=score,
                semantic_label=semantic_label,
            )
            estimated_objects.append(dynamic_obj)

        return estimated_objects

    def _ground_truths_to_frame_ground_truth(
        self,
        ground_truths: List[Dict[str, Any]],
        unix_time: float,
        frame_name: str = "0",
    ) -> FrameGroundTruth:
        """Convert ground truth dicts to FrameGroundTruth instance.

        Args:
            ground_truths: List of ground truth dicts with keys:
                - bbox_3d: [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, vx, vy]
                  (Same format as mmdet3d LiDARInstance3DBoxes)
                - label: int (class index)
                - num_lidar_pts: int (optional, number of lidar points)
            unix_time: Unix timestamp for the frame.
            frame_name: Name/ID of the frame.

        Returns:
            FrameGroundTruth instance.
        """
        gt_objects = []
        for gt in ground_truths:
            bbox = gt.get("bbox_3d", [])
            if len(bbox) < 7:
                continue

            # Extract bbox components
            # mmdet3d LiDARInstance3DBoxes format: [x, y, z, l, w, h, yaw, vx, vy]
            # where l=length, w=width, h=height
            x, y, z = bbox[0], bbox[1], bbox[2]
            l, w, h = bbox[3], bbox[4], bbox[5]
            yaw = bbox[6]

            # Velocity (optional)
            vx = bbox[7] if len(bbox) > 7 else 0.0
            vy = bbox[8] if len(bbox) > 8 else 0.0

            # Create quaternion from yaw
            orientation = Quaternion(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))

            # Get label
            label_idx = gt.get("label", 0)
            semantic_label = self._convert_index_to_label(int(label_idx))

            # Get point count (optional)
            num_pts = gt.get("num_lidar_pts", 0)

            # Shape size follows autoware_perception_evaluation convention: (length, width, height)
            dynamic_obj = DynamicObject(
                unix_time=unix_time,
                frame_id=self.frame_id,
                position=(x, y, z),
                orientation=orientation,
                shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(l, w, h)),
                velocity=(vx, vy, 0.0),
                semantic_score=1.0,  # GT always has score 1.0
                semantic_label=semantic_label,
                pointcloud_num=int(num_pts),
            )
            gt_objects.append(dynamic_obj)

        return FrameGroundTruth(
            unix_time=unix_time,
            frame_name=frame_name,
            objects=gt_objects,
            transforms=None,
            raw_data=None,
        )

    def add_frame(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        frame_name: Optional[str] = None,
    ) -> None:
        """Add a frame of predictions and ground truths for evaluation.

        Args:
            predictions: List of prediction dicts with keys:
                - bbox_3d: [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, vx, vy]
                - label: int (class index)
                - score: float (confidence score)
            ground_truths: List of ground truth dicts with keys:
                - bbox_3d: [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, vx, vy]
                - label: int (class index)
                - num_lidar_pts: int (optional)
            frame_name: Optional name for the frame.
        """
        if self.evaluator is None:
            self.reset()

        unix_time = time.time()
        if frame_name is None:
            frame_name = str(self._frame_count)

        # Convert predictions to DynamicObject
        estimated_objects = self._predictions_to_dynamic_objects(predictions, unix_time)

        # Convert ground truths to FrameGroundTruth
        frame_ground_truth = self._ground_truths_to_frame_ground_truth(ground_truths, unix_time, frame_name)

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
                - mAP_center_distance_bev_0.5
                - mAP_center_distance_bev_1.0
                - mAPH_center_distance_bev_0.5
                - car_AP_center_distance_bev_0.5
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

            # Add mAP and mAPH values
            map_key = f"mAP_{matching_mode}"
            maph_key = f"mAPH_{matching_mode}"
            metric_dict[map_key] = map_instance.map
            metric_dict[maph_key] = map_instance.maph

        return metric_dict

    def get_summary(self) -> DetectionSummary:
        """Get a summary of the evaluation including mAP and per-class metrics."""
        metrics = self.compute_metrics()

        # Extract primary metrics (first mAP value found)
        primary_map = None
        primary_maph = None
        per_class_ap = {}

        for key, value in metrics.items():
            if key.startswith("mAP_") and primary_map is None:
                primary_map = value
            elif key.startswith("mAPH_") and primary_maph is None:
                primary_maph = value
            elif "_AP_" in key and not key.startswith("mAP"):
                # Extract class name from key
                parts = key.split("_AP_")
                if len(parts) == 2:
                    class_name = parts[0]
                    if class_name not in per_class_ap:
                        per_class_ap[class_name] = value

        return DetectionSummary(
            mAP=primary_map or 0.0,
            mAPH=primary_maph or 0.0,
            per_class_ap=per_class_ap,
            num_frames=self._frame_count,
            detailed_metrics=metrics,
        )
