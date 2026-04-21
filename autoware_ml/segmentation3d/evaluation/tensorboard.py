# Copyright (c) TIER IV, Inc. All rights reserved.
"""Shared TensorBoard naming for T4 segmentation metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from .functional.t4_seg_eval import plot_confusion_matrix, range_label

_SUMMARY_KEYS = ("miou", "acc", "acc_cls", "mprecision", "mrecall", "mf1")


def build_t4_seg_tb_scalars(
    metrics: Dict[str, float],
    class_names: List[str],
    stage: str,
    distance_ranges: Optional[Iterable[Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Map canonical metric keys to the shared TensorBoard naming scheme."""
    tb_scalars: Dict[str, float] = {}

    for key in _SUMMARY_KEYS:
        if key in metrics:
            tb_scalars[f"{stage}/{key}"] = metrics[key]

    for class_name in class_names:
        if class_name in metrics:
            tb_scalars[f"{stage}/class_iou/{class_name}"] = metrics[class_name]
        for sub in ("precision", "recall", "f1"):
            metric_key = f"{sub}/{class_name}"
            if metric_key in metrics:
                tb_scalars[f"{stage}/class_{sub}/{class_name}"] = metrics[metric_key]

    for lo, hi in distance_ranges or []:
        bucket = range_label(lo, hi)
        for key in _SUMMARY_KEYS:
            metric_key = f"{bucket}/{key}"
            if metric_key in metrics:
                tb_scalars[f"{stage}/range/{bucket}/{key}"] = metrics[metric_key]
        for class_name in class_names:
            metric_key = f"{bucket}/{class_name}"
            if metric_key in metrics:
                tb_scalars[f"{stage}/range/{bucket}/class_iou/{class_name}"] = metrics[metric_key]
            for sub in ("precision", "recall", "f1"):
                metric_key = f"{bucket}/{sub}/{class_name}"
                if metric_key in metrics:
                    tb_scalars[f"{stage}/range/{bucket}/class_{sub}/{class_name}"] = metrics[metric_key]

    return tb_scalars


def iter_t4_seg_confusion_matrix_figures(eval_result, class_names: List[str], stage: str):
    """Yield standardised TensorBoard tags and matplotlib figures."""
    if eval_result.cm is not None and eval_result.cm.sum() > 0:
        yield f"{stage}/confusion_matrix", plot_confusion_matrix(eval_result.cm, class_names)

    for bucket, range_cm in eval_result.range_cms.items():
        if range_cm is None or range_cm.sum() == 0:
            continue
        tag = f"{stage}/confusion_matrix_{bucket.replace('-', '_').replace(' ', '_')}"
        yield tag, plot_confusion_matrix(range_cm, class_names, label=bucket)
