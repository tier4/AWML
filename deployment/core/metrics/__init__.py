"""
Unified Metrics Interfaces for AWML Deployment Framework.

This module provides task-specific metric interfaces that use autoware_perception_evaluation
as the single source of truth for metric computation. This ensures consistency between
training evaluation (T4MetricV2) and deployment evaluation.

Design Principles:
    1. 3D Detection → Detection3DMetricsInterface (mAP, mAPH using autoware_perception_eval)
    2. 2D Detection → Detection2DMetricsInterface (mAP using autoware_perception_eval, 2D mode)
    3. Classification → ClassificationMetricsInterface (accuracy, precision, recall, F1)

Usage:
    # For 3D detection (CenterPoint, etc.)
    from deployment.core.metrics import Detection3DMetricsInterface, Detection3DMetricsConfig

    config = Detection3DMetricsConfig(
        class_names=["car", "truck", "bus", "bicycle", "pedestrian"],
    )
    interface = Detection3DMetricsInterface(config)
    interface.add_frame(predictions, ground_truths)
    metrics = interface.compute_metrics()

    # For 2D detection (YOLOX, etc.)
    from deployment.core.metrics import Detection2DMetricsInterface, Detection2DMetricsConfig

    config = Detection2DMetricsConfig(
        class_names=["car", "truck", "bus", ...],
    )
    interface = Detection2DMetricsInterface(config)
    interface.add_frame(predictions, ground_truths)
    metrics = interface.compute_metrics()

    # For classification (Calibration, etc.)
    from deployment.core.metrics import ClassificationMetricsInterface, ClassificationMetricsConfig

    config = ClassificationMetricsConfig(
        class_names=["miscalibrated", "calibrated"],
    )
    interface = ClassificationMetricsInterface(config)
    interface.add_frame(prediction_label, ground_truth_label, probabilities)
    metrics = interface.compute_metrics()
"""

from deployment.core.metrics.base_metrics_interface import (
    BaseMetricsConfig,
    BaseMetricsInterface,
    ClassificationSummary,
    DetectionSummary,
)
from deployment.core.metrics.classification_metrics import (
    ClassificationMetricsConfig,
    ClassificationMetricsInterface,
)
from deployment.core.metrics.detection_2d_metrics import (
    Detection2DMetricsConfig,
    Detection2DMetricsInterface,
)
from deployment.core.metrics.detection_3d_metrics import (
    Detection3DMetricsConfig,
    Detection3DMetricsInterface,
)

__all__ = [
    # Base classes
    "BaseMetricsInterface",
    "BaseMetricsConfig",
    "ClassificationSummary",
    "DetectionSummary",
    # 3D Detection
    "Detection3DMetricsInterface",
    "Detection3DMetricsConfig",
    # 2D Detection
    "Detection2DMetricsInterface",
    "Detection2DMetricsConfig",
    # Classification
    "ClassificationMetricsInterface",
    "ClassificationMetricsConfig",
]
