"""
3D Detection Metrics Interface using autoware_perception_evaluation.

This module provides an interface to compute 3D detection metrics (mAP, mAPH)
using autoware_perception_evaluation, ensuring consistent metrics between
training evaluation (T4MetricV2) and deployment evaluation.

Like T4MetricV2, multiple evaluators are created for different distance ranges. Frames are
buffered once and replayed into each range's evaluator at compute time (see
``BaseMetricsInterface``), so peak memory is independent of the number of ranges.

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
    # Returns: {"bev_center_0.0-121.0_mAP_center_distance_bev": 0.7, ...}
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from mmengine.config import Config, ConfigDict
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.object import DynamicObject
from perception_eval.common.shape import Shape, ShapeType
from perception_eval.evaluation.metrics import MetricsScore
from pyquaternion import Quaternion

from deployment.metrics.base_metrics_interface import BaseMetricsConfig
from deployment.metrics.detection_base import DetectionMetricsInterface

logger = logging.getLogger(__name__)

# Prefix for the per-distance-range evaluator names (and thus metric-key prefixes).
_RANGE_FILTER_NAME = "bev_center"


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
        if self.evaluation_config_dict is None:
            object.__setattr__(
                self,
                "evaluation_config_dict",
                {
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
                },
            )

        if self.critical_object_filter_config is None:
            num_classes = len(self.class_names)
            object.__setattr__(
                self,
                "critical_object_filter_config",
                {
                    "target_labels": self.class_names,
                    "ignore_attributes": None,
                    "max_distance_list": [121.0] * num_classes,
                    "min_distance_list": [-121.0] * num_classes,
                },
            )

        if self.frame_pass_fail_config is None:
            num_classes = len(self.class_names)
            object.__setattr__(
                self,
                "frame_pass_fail_config",
                {
                    "target_labels": self.class_names,
                    "matching_threshold_list": [2.0] * num_classes,
                    "confidence_threshold_list": None,
                },
            )


_T4METRIC_V2_EVALUATOR_TYPE = "T4MetricV2"


def extract_t4metric_v2_config(model_cfg: Config) -> Detection3DMetricsConfig:
    """Build a `Detection3DMetricsConfig` from an MMEngine model config.

    This is shared by every 3D detection deployment project: the metrics are
    derived from the model config's `T4MetricV2` val evaluator, not from the
    model architecture, so projects reuse this instead of reimplementing it.

    Args:
        model_cfg: MMEngine model configuration.

    Returns:
        Detection3DMetricsConfig instance with extracted settings.

    Raises:
        ValueError: If class_names not provided and not found in model_cfg,
                   or if evaluator config is missing or not T4MetricV2 type.
    """

    def read_required_cfg_value(cfg: Config | ConfigDict, key: str) -> Any:
        """Read a required key/attribute from config object.

        Args:
            cfg: MMEngine Config or ConfigDict to read from.
            key: Required key/attribute name.

        Returns:
            Value stored at the given key/attribute.

        Raises:
            ValueError: If key/attribute does not exist in cfg.
        """
        if key in cfg:
            return cfg[key]
        if hasattr(cfg, key):
            return getattr(cfg, key)
        raise ValueError(f"Missing required key/attribute '{key}'")

    class_names = read_required_cfg_value(model_cfg, "class_names")
    evaluator_cfg = read_required_cfg_value(model_cfg, "val_evaluator")

    evaluator_type = read_required_cfg_value(evaluator_cfg, "type")
    if evaluator_type != _T4METRIC_V2_EVALUATOR_TYPE:
        raise ValueError(f"Evaluator type is '{evaluator_type}', not '{_T4METRIC_V2_EVALUATOR_TYPE}'")

    perception_configs = read_required_cfg_value(evaluator_cfg, "perception_evaluator_configs")
    evaluation_config_dict = read_required_cfg_value(perception_configs, "evaluation_config_dict")
    frame_id = read_required_cfg_value(perception_configs, "frame_id")

    critical_object_filter_config = read_required_cfg_value(evaluator_cfg, "critical_object_filter_config")
    frame_pass_fail_config = read_required_cfg_value(evaluator_cfg, "frame_pass_fail_config")

    return Detection3DMetricsConfig(
        class_names=class_names,
        frame_id=frame_id,
        evaluation_config_dict=evaluation_config_dict,
        critical_object_filter_config=critical_object_filter_config,
        frame_pass_fail_config=frame_pass_fail_config,
    )


class Detection3DMetricsInterface(DetectionMetricsInterface):
    # TODO(vividf): refactor this class after refactoring T4MetricV2
    """
    Interface for computing 3D detection metrics using autoware_perception_evaluation.

    Computes mAP, mAPH, and other detection metrics consistent with the T4MetricV2 used
    during training, evaluated over multiple distance ranges.
    """

    _supports_aph = True

    def __init__(
        self,
        config: Detection3DMetricsConfig,
        data_root: str = "data/t4dataset/",
        result_root_directory: str = "/tmp/perception_eval/",
    ) -> None:
        """
        Initialize the 3D detection metrics interface.

        Args:
            config: Configuration for 3D detection metrics.
            data_root: Root directory of the dataset.
            result_root_directory: Directory for saving evaluation results.
        """
        super().__init__(config)
        self.config: Detection3DMetricsConfig = config

        self._bev_distance_ranges = self._resolve_distance_ranges(config.evaluation_config_dict)
        self._create_evaluator_specs(config, data_root, result_root_directory)

    @staticmethod
    def _resolve_distance_ranges(cfg_dict: Optional[Mapping[str, Any]]) -> List[tuple]:
        """Validate and expand min/max distance into a list of (min, max) ranges."""
        if cfg_dict is None:
            cfg_dict = {}
        if not isinstance(cfg_dict, Mapping):
            raise TypeError(f"evaluation_config_dict must be a dict, got {type(cfg_dict).__name__}")

        min_distance = cfg_dict.get("min_distance")
        max_distance = cfg_dict.get("max_distance")

        if isinstance(min_distance, (int, float)) and isinstance(max_distance, (int, float)):
            min_distance = [float(min_distance)]
            max_distance = [float(max_distance)]
        elif not isinstance(min_distance, list) or not isinstance(max_distance, list):
            raise ValueError(
                "min_distance and max_distance must be either scalars (int/float) or lists for multi-evaluator mode. "
                f"Got min_distance={type(min_distance)}, max_distance={type(max_distance)}"
            )

        if len(min_distance) != len(max_distance):
            raise ValueError(
                "min_distance and max_distance must have the same length. "
                f"Got len(min_distance)={len(min_distance)}, len(max_distance)={len(max_distance)}"
            )

        if not min_distance or not max_distance:
            raise ValueError("min_distance and max_distance lists cannot be empty")

        return list(zip(min_distance, max_distance))

    def _create_evaluator_specs(
        self,
        config: Detection3DMetricsConfig,
        data_root: str,
        result_root_directory: str,
    ) -> None:
        """Create one evaluator spec per distance range (like T4MetricV2)."""
        base_eval_config = config.evaluation_config_dict
        if base_eval_config is None:
            base_eval_config = {}
        if not isinstance(base_eval_config, Mapping):
            raise TypeError(f"evaluation_config_dict must be a dict, got {type(base_eval_config).__name__}")

        for min_dist, max_dist in self._bev_distance_ranges:
            eval_config_dict = dict(base_eval_config)
            eval_config_dict["min_distance"] = min_dist
            eval_config_dict["max_distance"] = max_dist

            name = f"{_RANGE_FILTER_NAME}_{min_dist}-{max_dist}"
            self._evaluator_specs[name] = self._build_evaluator_spec(
                eval_config_dict,
                data_root=data_root,
                result_root_directory=result_root_directory,
                critical_object_filter_config=config.critical_object_filter_config,
                frame_pass_fail_config=config.frame_pass_fail_config,
            )

    def _to_dynamic_objects_3d(
        self,
        entries: List[Dict[str, Any]],
        unix_time: float,
        *,
        is_gt: bool,
    ) -> List[DynamicObject]:
        """Convert prediction/ground-truth dicts to DynamicObject instances.

        Args:
            entries: List of dicts with keys:
                - bbox_3d: [x, y, z, l, w, h, yaw] or [x, y, z, l, w, h, yaw, vx, vy]
                  (Same format as mmdet3d LiDARInstance3DBoxes)
                - label: int (class index)
                - score: float (confidence; ignored for ground truth, which is always 1.0)
                - num_lidar_pts: int (ground truth only, optional)
            unix_time: Unix timestamp for the frame.
            is_gt: Whether ``entries`` are ground truths (forces score 1.0 and reads point counts).

        Returns:
            List of DynamicObject instances.
        """
        kind = "ground truth" if is_gt else "prediction"

        objects: List[DynamicObject] = []
        for entry in entries:
            bbox = entry.get("bbox_3d", [])
            if len(bbox) < 7:
                continue

            # mmdet3d LiDARInstance3DBoxes format: [x, y, z, l, w, h, yaw, vx, vy]
            # where l=length, w=width, h=height.
            x, y, z = bbox[0], bbox[1], bbox[2]
            l, w, h = bbox[3], bbox[4], bbox[5]
            yaw = bbox[6]

            # Skip non-finite boxes: a NaN/inf yaw produces a NaN quaternion that
            # silently corrupts matching across the whole frame.
            if not np.all(np.isfinite([x, y, z, l, w, h, yaw])):
                logger.warning("Skipping %s with non-finite bbox_3d: %s", kind, bbox)
                continue

            vx = bbox[7] if len(bbox) > 7 else 0.0
            vy = bbox[8] if len(bbox) > 8 else 0.0

            orientation = Quaternion(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
            semantic_label = self._convert_index_to_label(int(entry.get("label", 0)))
            score = 1.0 if is_gt else float(entry.get("score", 0.0))

            kwargs: Dict[str, Any] = {}
            if is_gt:
                kwargs["pointcloud_num"] = int(entry.get("num_lidar_pts", 0))

            # Shape size follows autoware_perception_evaluation convention: (length, width, height)
            objects.append(
                DynamicObject(
                    unix_time=unix_time,
                    frame_id=self.frame_id,
                    position=(x, y, z),
                    orientation=orientation,
                    shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(l, w, h)),
                    velocity=(vx, vy, 0.0),
                    semantic_score=score,
                    semantic_label=semantic_label,
                    **kwargs,
                )
            )

        return objects

    def add_frame(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        frame_name: Optional[str] = None,
    ) -> None:
        """Buffer a frame of predictions and ground truths for evaluation.

        Args:
            predictions: List of prediction dicts with keys bbox_3d, label, score.
            ground_truths: List of ground truth dicts with keys bbox_3d, label, num_lidar_pts (optional).
            frame_name: Optional name for the frame.
        """
        unix_time = time.time()
        if frame_name is None:
            frame_name = str(self._frame_count)

        estimated_objects = self._to_dynamic_objects_3d(predictions, unix_time, is_gt=False)
        gt_objects = self._to_dynamic_objects_3d(ground_truths, unix_time, is_gt=True)

        frame_ground_truth = FrameGroundTruth(
            unix_time=unix_time,
            frame_name=frame_name,
            objects=gt_objects,
            transforms=None,
            raw_data=None,
        )

        self._buffer_frame(unix_time, estimated_objects, frame_ground_truth)

    def format_metrics_report(self) -> str:
        """Format the metrics report for all distance ranges as a human-readable string.

        Uses cached scene results from ``compute_metrics`` if available.
        """
        if not self._last_scene_results:
            self.compute_metrics()

        reports = []
        for eval_name, metrics_score in self._last_scene_results.items():
            distance_range = eval_name.replace(f"{_RANGE_FILTER_NAME}_", "")
            reports.append(
                f"\n{'=' * 80}\n" f"Distance Range: {distance_range} m\n" f"{'=' * 80}\n" f"{metrics_score}"
            )

        if not reports:
            raise RuntimeError("Failed to generate metrics report. Ensure that metrics have been computed.")

        return "\n".join(reports)

    def _select_summary_score(self) -> Optional[MetricsScore]:
        """Summarize from the last (widest) distance bucket's score."""
        if not self._bev_distance_ranges:
            return None
        last_min_dist, last_max_dist = self._bev_distance_ranges[-1]
        return self._last_scene_results.get(f"{_RANGE_FILTER_NAME}_{last_min_dist}-{last_max_dist}")
