# Copyright (c) TIER IV, Inc. All rights reserved.
"""Helpers for 3D semantic segmentation evaluation."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable

_EPS = 1e-10


def fast_hist(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Confusion matrix for one sample (matches mmdet3d ``fast_hist``).

    ``hist[gt_class, pred_class]`` = number of points.

    Args:
        preds: Predicted label array, shape ``(N,)``.
        labels: Ground-truth label array, shape ``(N,)``.
        num_classes: Number of classes.

    Returns:
        ``np.ndarray`` of shape ``(num_classes, num_classes)``.
    """
    k = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
    bin_count = np.bincount(
        num_classes * labels[k].astype(int) + preds[k],
        minlength=num_classes**2,
    )
    return bin_count[: num_classes**2].reshape(num_classes, num_classes)


def per_class_iou(hist: np.ndarray) -> np.ndarray:
    """Per-class IoU from cumulative confusion matrix."""
    tp = np.diag(hist)
    denom = hist.sum(1) + hist.sum(0) - tp
    return np.where(denom > _EPS, tp / (denom + _EPS), np.nan)


def get_acc(hist: np.ndarray) -> float:
    """Overall point-level accuracy."""
    return float(np.diag(hist).sum() / (hist.sum() + _EPS))


def get_acc_cls(hist: np.ndarray) -> float:
    """Class-average accuracy (same as macro recall)."""
    return float(np.nanmean(np.diag(hist) / (hist.sum(axis=1) + _EPS)))


def per_class_precision(hist: np.ndarray) -> np.ndarray:
    """Per-class precision: TP / (TP + FP) = TP / predicted-as-class."""
    tp = np.diag(hist)
    predicted = hist.sum(axis=0)  # column sums
    return np.where(predicted > _EPS, tp / (predicted + _EPS), np.nan)


def per_class_recall(hist: np.ndarray) -> np.ndarray:
    """Per-class recall: TP / (TP + FN) = TP / actual-class-count."""
    tp = np.diag(hist)
    actual = hist.sum(axis=1)  # row sums
    return np.where(actual > _EPS, tp / (actual + _EPS), np.nan)


def per_class_f1(hist: np.ndarray) -> np.ndarray:
    """Per-class F1 score: 2 * precision * recall / (precision + recall)."""
    prec = per_class_precision(hist)
    rec = per_class_recall(hist)
    denom = prec + rec
    return np.where(denom > _EPS, 2.0 * prec * rec / (denom + _EPS), np.nan)


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """Row-normalise so each row sums to 1 (GT-class perspective).

    Rows without any GT sample are set to 0 rather than NaN so the result
    can be safely passed to matplotlib's ``imshow``.
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    return cm / safe


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    label: str = "",
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Render a confusion matrix as a matplotlib Figure for TensorBoard.

    * Y-axis = "True label", X-axis = "Predicted label".
    * Color scale fixed at ``[0, 1]`` (normalised fractions) so plots from
      different epochs are directly comparable.
    * Numeric value annotated in every cell.

    Args:
        cm: ``(num_classes, num_classes)`` confusion matrix, ``cm[gt][pred]``.
        class_names: Human-readable class names.
        normalize: If ``True`` (default), row-normalise before plotting.
        label: Optional range label appended to the figure title.

    Returns:
        ``matplotlib.figure.Figure`` - caller is responsible for closing it.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize as MplNormalize

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")

    nc = cm.shape[0]
    cm_plot = normalize_confusion_matrix(cm) if normalize else cm.astype(float)

    fig, ax = plt.subplots(figsize=(max(10, nc * 0.6), max(8, nc * 0.55)))
    im = ax.imshow(
        cm_plot,
        interpolation="nearest",
        cmap="Blues",
        norm=MplNormalize(vmin=0.0, vmax=1.0),
    )
    fig.colorbar(im, ax=ax, shrink=0.8)

    font_size = max(4, 7 - nc // 10)
    for i in range(nc):
        for j in range(nc):
            val = cm_plot[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=font_size, color=color)

    title = "Confusion Matrix"
    if label:
        title += f" [{label}]"
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)

    tick_marks = np.arange(nc)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    fig.tight_layout()
    return fig


def figure_to_numpy(fig) -> np.ndarray:
    """Convert a matplotlib Figure to a uint8 HWC NumPy array (RGB).

    Uses the in-memory PNG path; does not require a display.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    from PIL import Image  # lazy import; PIL is a lightweight dep

    img = Image.open(buf).convert("RGB")
    return np.array(img)


def compute_bev_distance(coords: np.ndarray) -> np.ndarray:
    """BEV distance from ego: ``sqrt(x^2 + y^2)`` for each point.

    Args:
        coords: ``(N, ≥2)`` array (first two columns are X, Y in metres).

    Returns:
        ``(N,)`` array of distances in metres.
    """
    return np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)


def range_label(lo: float, hi: float) -> str:
    """Human-readable range label, e.g. ``'0-20m'``."""
    return f"{lo:g}-{hi:g}m"


@dataclass
class SegEvalResult:
    """Evaluation result with scalar metrics and raw confusion matrices.

    Attributes:
        metrics: Flat dict of scalar metrics keyed in mmdetection3d style:
            ``miou``, ``acc``, ``acc_cls``, per-class IoU by name,
            ``mprecision``, ``mrecall``, ``mf1``,
            ``precision/{class}``, ``recall/{class}``, ``f1/{class}``; and
            for each range bucket, the same keys prefixed with
            ``{range_label}/`` (e.g. ``0-20m/miou``).
        cm: Total confusion matrix ``(num_classes, num_classes)``.
        range_cms: Per-range confusion matrices keyed by range label.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    cm: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    range_cms: Dict[str, np.ndarray] = field(default_factory=dict)


def _compute_bucket_metrics(
    hist: np.ndarray,
    label2cat: Dict[int, str],
    ignore_index: int,
    prefix: str,
) -> Dict[str, float]:
    """Derive all scalar metrics from a confusion histogram.

    Args:
        hist: ``(num_classes, num_classes)`` cumulative confusion matrix.
        label2cat: ``{index: class_name}`` mapping.
        ignore_index: Class index to exclude from averages.
        prefix: String prepended to every key (e.g. ``'0-20m/'``).

    Returns:
        Flat dict of scalar metrics for this bucket.
    """
    num_classes = hist.shape[0]
    out: Dict[str, float] = {}

    # Per-class IoU - identical to mmdet3d seg_eval
    iou = per_class_iou(hist)
    if 0 <= ignore_index < num_classes:
        iou[ignore_index] = np.nan
    miou = float(np.nanmean(iou))

    # Per-class precision / recall / F1
    prec = per_class_precision(hist)
    rec = per_class_recall(hist)
    f1 = per_class_f1(hist)
    if 0 <= ignore_index < num_classes:
        prec[ignore_index] = np.nan
        rec[ignore_index] = np.nan
        f1[ignore_index] = np.nan

    out[f"{prefix}miou"] = miou
    out[f"{prefix}acc"] = get_acc(hist)
    out[f"{prefix}acc_cls"] = get_acc_cls(hist)
    out[f"{prefix}mprecision"] = float(np.nanmean(prec))
    out[f"{prefix}mrecall"] = float(np.nanmean(rec))
    out[f"{prefix}mf1"] = float(np.nanmean(f1))

    for idx in range(num_classes):
        if idx == ignore_index:
            continue
        name = label2cat.get(idx, str(idx))
        out[f"{prefix}{name}"] = float(iou[idx])
        out[f"{prefix}precision/{name}"] = float(prec[idx])
        out[f"{prefix}recall/{name}"] = float(rec[idx])
        out[f"{prefix}f1/{name}"] = float(f1[idx])

    return out


def _print_bucket_table(
    hist: np.ndarray,
    label2cat: Dict[int, str],
    ignore_index: int,
    title: str,
    logger=None,
) -> None:
    """Print an AsciiTable for one evaluation bucket."""
    num_classes = hist.shape[0]
    iou = per_class_iou(hist)
    if 0 <= ignore_index < num_classes:
        iou[ignore_index] = np.nan
    prec = per_class_precision(hist)
    rec = per_class_recall(hist)
    f1 = per_class_f1(hist)

    header = ["class", "IoU", "Prec", "Rec", "F1"]
    rows = [header]
    for idx in range(num_classes):
        if idx == ignore_index:
            continue
        name = label2cat.get(idx, str(idx))
        rows.append(
            [
                name,
                f"{iou[idx]:.4f}" if not np.isnan(iou[idx]) else "N/A",
                f"{prec[idx]:.4f}" if not np.isnan(prec[idx]) else "N/A",
                f"{rec[idx]:.4f}" if not np.isnan(rec[idx]) else "N/A",
                f"{f1[idx]:.4f}" if not np.isnan(f1[idx]) else "N/A",
            ]
        )
    miou = float(np.nanmean(iou))
    mprec = float(np.nanmean(prec))
    mrec = float(np.nanmean(rec))
    mf1 = float(np.nanmean(f1))
    rows.append(["mean", f"{miou:.4f}", f"{mprec:.4f}", f"{mrec:.4f}", f"{mf1:.4f}"])
    rows.append(["acc", f"{get_acc(hist):.4f}", "-", "-", "-"])
    rows.append(["acc_cls", f"{get_acc_cls(hist):.4f}", "-", "-", "-"])

    table = AsciiTable(rows, title=title)
    table.inner_footing_row_border = True
    print_log("\n" + table.table, logger=logger)


def t4_seg_eval(
    gt_labels: List[np.ndarray],
    seg_preds: List[np.ndarray],
    label2cat: Dict[int, str],
    ignore_index: int,
    coords_list: Optional[List[Optional[np.ndarray]]] = None,
    distance_ranges: Optional[List[Tuple[float, float]]] = None,
    logger=None,
) -> SegEvalResult:
    """Semantic segmentation evaluation with optional range-based breakdown.

    Produces the same top-level keys as ``mmdet3d.evaluation.seg_eval``
    (``miou``, ``acc``, ``acc_cls``, per-class IoU by name) and additionally
    adds precision / recall / F1 metrics and optional per-range variants.

    Args:
        gt_labels: Ground-truth label arrays, one per sample.
        seg_preds: Predicted label arrays, one per sample.
        label2cat: ``{output_index: class_name}`` mapping.
        ignore_index: Label to exclude from metric computation.
        coords_list: Optional per-sample XYZ coordinate arrays ``(N, ≥2)``.
            When provided together with ``distance_ranges``, range-based
            metrics are computed; otherwise only total metrics are returned.
        distance_ranges: List of ``(lo, hi)`` metre pairs, e.g.
            ``[(0, 20), (20, 40), ..., (100, 120)]``.
        logger: Optional logger for tabular output.

    Returns:
        :class:`SegEvalResult` with scalar metrics dict, total CM, and
        per-range CMs.
    """
    assert len(gt_labels) == len(seg_preds), (
        f"gt and pred lists must have the same length " f"({len(gt_labels)} vs {len(seg_preds)})"
    )

    num_classes = len(label2cat)
    use_ranges = bool(distance_ranges and coords_list is not None)

    total_hist = np.zeros((num_classes, num_classes), dtype=np.float64)

    if use_ranges:
        range_hists: Dict[str, np.ndarray] = {
            range_label(lo, hi): np.zeros((num_classes, num_classes), dtype=np.float64)
            for lo, hi in distance_ranges  # type: ignore[union-attr]
        }
    else:
        range_hists = {}

    for i in range(len(gt_labels)):
        gt = gt_labels[i].astype(np.int64)
        pred = seg_preds[i].astype(np.int64)

        pred[gt == ignore_index] = ignore_index
        gt[gt == ignore_index] = ignore_index

        h = fast_hist(pred, gt, num_classes)
        total_hist += h

        if use_ranges:
            assert coords_list is not None
            coord = coords_list[i]
            if coord is None:
                continue
            coord = np.asarray(coord)
            if coord.ndim != 2 or coord.shape[1] < 2 or coord.shape[0] != gt.size:
                continue
            dist = compute_bev_distance(coord)
            for lo, hi in distance_ranges:  # type: ignore[union-attr]
                lbl = range_label(lo, hi)
                mask = (dist >= lo) & (dist < hi)
                if not np.any(mask):
                    continue
                h_r = fast_hist(pred[mask], gt[mask], num_classes)
                range_hists[lbl] += h_r

    _print_bucket_table(total_hist, label2cat, ignore_index, title="Total", logger=logger)
    metrics = _compute_bucket_metrics(total_hist, label2cat, ignore_index, prefix="")

    for lbl, hist_r in range_hists.items():
        if hist_r.sum() == 0:
            continue
        _print_bucket_table(hist_r, label2cat, ignore_index, title=lbl, logger=logger)
        metrics.update(_compute_bucket_metrics(hist_r, label2cat, ignore_index, prefix=f"{lbl}/"))

    return SegEvalResult(metrics=metrics, cm=total_hist, range_cms=range_hists)
