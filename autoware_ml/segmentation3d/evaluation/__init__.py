# Copyright (c) TIER IV, Inc. All rights reserved.
"""Segmentation evaluation: functional helpers + MMEngine metric adapter."""

from .functional.t4_seg_eval import (
    SegEvalResult,
    compute_bev_distance,
    fast_hist,
    figure_to_numpy,
    get_acc,
    get_acc_cls,
    normalize_confusion_matrix,
    per_class_f1,
    per_class_iou,
    per_class_precision,
    per_class_recall,
    plot_confusion_matrix,
    range_label,
    t4_seg_eval,
)
from .metrics.t4_seg_metric import T4SegMetric

__all__ = [
    "SegEvalResult",
    "T4SegMetric",
    "compute_bev_distance",
    "fast_hist",
    "figure_to_numpy",
    "get_acc",
    "get_acc_cls",
    "normalize_confusion_matrix",
    "per_class_f1",
    "per_class_iou",
    "per_class_precision",
    "per_class_recall",
    "plot_confusion_matrix",
    "range_label",
    "t4_seg_eval",
]
