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
    t4_seg_eval_from_hists,
    update_seg_eval_histograms,
)
from .metrics.t4_seg_metric import T4SegMetric
from .tensorboard import build_t4_seg_tb_scalars, iter_t4_seg_confusion_matrix_figures

__all__ = [
    "SegEvalResult",
    "T4SegMetric",
    "build_t4_seg_tb_scalars",
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
    "t4_seg_eval_from_hists",
    "iter_t4_seg_confusion_matrix_figures",
    "update_seg_eval_histograms",
]
