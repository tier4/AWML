# Copyright (c) TIER IV, Inc. All rights reserved.
"""MMEngine metric adapter for shared T4 segmentation evaluation."""

import os.path as osp
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import mmcv
import numpy as np
from mmdet3d.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from autoware_ml.segmentation3d.evaluation.functional.t4_seg_eval import (
    figure_to_numpy,
    plot_confusion_matrix,
    range_label,
    t4_seg_eval,
)


@METRICS.register_module()
class T4SegMetric(BaseMetric):
    """3D semantic segmentation evaluation metric for T4 datasets.

    Parameters
    ----------
    num_classes:
        Number of output classes (excluding the ignore class).
    ignore_index:
        Label value to skip during evaluation.  Defaults to the value set in
        ``dataset_meta``; the explicit argument takes priority.
    distance_ranges:
        Optional list of ``(lo, hi)`` metre pairs for range-based breakdown,
        e.g. ``[(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120)]``.
    collect_device:
        Device used for collecting results across ranks. ``'cpu'`` or ``'gpu'``.
    prefix:
        Optional metric-name prefix.
    pklfile_prefix:
        If set, raw predictions are written to ``{pklfile_prefix}.pkl``.
    submission_prefix:
        If set, predictions are exported in ScanNet TXT format to this path
        instead of computing metrics.
    """

    default_prefix: Optional[str] = None

    def __init__(
        self,
        num_classes: Optional[int] = None,
        ignore_index: Optional[int] = None,
        distance_ranges: Optional[List[Tuple[float, float]]] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        pklfile_prefix: Optional[str] = None,
        submission_prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(prefix=prefix, collect_device=collect_device)
        self._num_classes = num_classes
        self._ignore_index = ignore_index
        self.distance_ranges = distance_ranges or []
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        # Counter used as the TensorBoard global-step for CM images.
        self._eval_step: int = 0

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Collect one batch of model outputs for later aggregation."""
        batch_coords = self._extract_batch_coords(data_batch, data_samples)

        for i, data_sample in enumerate(data_samples):
            pred_field = data_sample.get("pred_pts_seg", {})
            ann_field = data_sample.get("eval_ann_info", {})

            pred = self._to_numpy(pred_field.get("pts_semantic_mask"))
            gt = self._to_numpy(ann_field.get("pts_semantic_mask"))

            if pred is None or gt is None or pred.size != gt.size:
                continue

            coord_i = batch_coords[i] if batch_coords else None
            if coord_i is not None:
                if coord_i.shape[0] > gt.size:
                    coord_i = coord_i[: gt.size]
                elif coord_i.shape[0] < gt.size:
                    coord_i = None

            self.results.append(
                dict(
                    pred=pred,
                    gt=gt,
                    coord=coord_i,
                    # Keep original annotation info for submission export.
                    eval_ann_info=ann_field,
                )
            )

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Aggregate per-batch results and return the full metrics dict."""
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return {}

        if not results:
            logger.warning("T4SegMetric: no results to evaluate.")
            return {}

        ignore_index = self._get_ignore_index()
        label2cat = self._get_label2cat()

        target_num_classes = self._num_classes or len(label2cat)
        target_num_classes = max(target_num_classes, ignore_index + 1)
        for idx in range(target_num_classes):
            if idx not in label2cat:
                label2cat[idx] = "ignore" if idx == ignore_index else str(idx)

        gt_labels = [r["gt"] for r in results]
        seg_preds = [r["pred"] for r in results]
        coords_list = [r.get("coord") for r in results] if self.distance_ranges else None
        if self.distance_ranges and (not coords_list or all(c is None for c in coords_list)):
            logger.warning(
                "T4SegMetric: distance_ranges is configured but no coordinates "
                "were extracted from data_batch. Range-based confusion matrices "
                "will be empty."
            )

        eval_result = t4_seg_eval(
            gt_labels,
            seg_preds,
            label2cat,
            ignore_index,
            coords_list=coords_list,
            distance_ranges=self.distance_ranges if self.distance_ranges else None,
            logger=logger,
        )

        if self.distance_ranges and eval_result.cm.sum() > 0:
            covered = sum(cm.sum() for cm in eval_result.range_cms.values())
            if covered == 0:
                logger.warning(
                    "T4SegMetric: total confusion matrix is non-empty but all "
                    "range-based confusion matrices are empty. This usually "
                    "means distance_ranges do not cover observed distances or "
                    "coordinate extraction is still misaligned."
                )

        self._log_confusion_matrix_images(eval_result, label2cat)
        self._eval_step += 1

        return eval_result.metrics

    def format_results(self, results: list) -> None:
        """Export predictions to TXT files for submission (ScanNet format)."""
        submission_prefix = self.submission_prefix
        if submission_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            submission_prefix = osp.join(tmp_dir.name, "results")
        mmcv.mkdir_or_exist(submission_prefix)

        ignore_index_val = self._get_ignore_index()
        label2cat_map = self._get_label2cat()
        num_labels = len(label2cat_map)

        cat2label = np.zeros(num_labels, dtype=np.int64)
        for out_idx, _ in label2cat_map.items():
            if out_idx != ignore_index_val:
                cat2label[out_idx] = out_idx

        meta = getattr(self, "dataset_meta", {}) or {}
        if "label2cat" in meta:
            for original_label, output_idx in meta["label2cat"].items():
                if isinstance(output_idx, int) and output_idx != ignore_index_val:
                    cat2label[output_idx] = original_label

        for r in results:
            ann = r.get("eval_ann_info", {})
            sample_idx = (ann.get("point_cloud") or {}).get("lidar_idx", "unknown")
            pred_sem = r["pred"].astype(np.int64)
            pred_label = cat2label[pred_sem]
            curr_file = f"{submission_prefix}/{sample_idx}.txt"
            np.savetxt(curr_file, pred_label, fmt="%d")

    @staticmethod
    def _to_numpy(v) -> Optional[np.ndarray]:
        """Convert tensor / array-like to a flat int64 numpy array."""
        if v is None:
            return None
        if hasattr(v, "cpu"):
            v = v.cpu().numpy()
        arr = np.asarray(v, dtype=np.int64)
        return arr.ravel()

    @staticmethod
    def _extract_batch_coords(data_batch: dict, data_samples: Sequence[dict]) -> Optional[List]:
        """Try to extract XY coordinates from packed input points.

        Returns a list of length ``len(data_samples)`` where each entry is either a
        ``(N, 2)`` float32 array or ``None``.
        """
        try:
            n_samples = len(data_samples)

            def _unwrap_points_tensor(obj):
                """Best-effort unwrapping for collate/data wrappers."""
                cur = obj
                for _ in range(8):
                    if cur is None:
                        return None
                    if hasattr(cur, "tensor"):
                        cur = cur.tensor
                        continue
                    if hasattr(cur, "data") and not isinstance(cur, np.ndarray):
                        nxt = getattr(cur, "data")
                        if nxt is cur:
                            break
                        cur = nxt
                        continue
                    if isinstance(cur, (list, tuple)) and len(cur) == 1:
                        cur = cur[0]
                        continue
                    break
                return cur

            inputs = data_batch.get("inputs") or {}
            if not isinstance(inputs, dict):
                inputs = {}
            points_data = inputs.get("points")
            if points_data is None:
                return None

            num_points_list = []
            for ds in data_samples:
                meta = getattr(ds, "metainfo", {}) or {}
                n = meta.get("num_points", None)
                num_points_list.append(int(n) if isinstance(n, (int, np.integer)) else None)

            if not isinstance(points_data, (list, tuple)):
                raw = _unwrap_points_tensor(points_data)
                if raw is not None and hasattr(raw, "cpu"):
                    raw = raw.cpu().numpy()
                raw_arr = np.asarray(raw) if raw is not None else None
                if (
                    raw_arr is not None
                    and raw_arr.ndim >= 2
                    and all(v is not None for v in num_points_list)
                    and sum(num_points_list) <= raw_arr.shape[0]
                ):
                    split = []
                    st = 0
                    for n in num_points_list:
                        ed = st + int(n)
                        split.append(raw_arr[st:ed])
                        st = ed
                    points_data = split
                else:
                    points_data = [points_data] * n_samples

            coords = []
            for pts in points_data[:n_samples]:
                if pts is None:
                    coords.append(None)
                    continue
                tens = _unwrap_points_tensor(pts)
                if tens is None:
                    coords.append(None)
                    continue
                if hasattr(tens, "cpu"):
                    tens = tens.cpu().numpy()
                arr = np.asarray(tens, dtype=np.float32)
                if arr.ndim >= 2 and arr.shape[1] >= 2:
                    coords.append(arr[:, :2])
                else:
                    coords.append(None)
            return coords
        except Exception:
            return None

    def _get_label2cat(self) -> Dict[int, str]:
        """Resolve {output_index: class_name} from constructor args or meta."""
        meta = getattr(self, "dataset_meta", {}) or {}
        label2cat = meta.get("label2cat")
        if isinstance(label2cat, dict):
            return {int(k): str(v) for k, v in label2cat.items()}
        # Fallback: use class_names list if available
        class_names = meta.get("classes") or meta.get("class_names")
        if isinstance(class_names, (list, tuple)):
            return {i: str(name) for i, name in enumerate(class_names)}
        # Last resort: numeric class names
        nc = self._num_classes or 1
        return {i: str(i) for i in range(nc)}

    def _get_ignore_index(self) -> int:
        if self._ignore_index is not None:
            return self._ignore_index
        meta = getattr(self, "dataset_meta", {}) or {}
        return int(meta.get("ignore_index", -1))

    def _log_confusion_matrix_images(self, eval_result, label2cat: Dict[int, str]) -> None:
        """Log normalised confusion-matrix images to TensorBoard (rank-0 only)."""
        try:
            from mmengine.visualization import Visualizer

            vis = Visualizer.get_current_instance()
        except Exception:
            return

        num_classes = int(eval_result.cm.shape[0]) if eval_result.cm is not None else len(label2cat)
        class_names = [label2cat.get(i, str(i)) for i in range(num_classes)]
        step = self._eval_step
        tag_prefix = f"{self.prefix}/" if self.prefix else ""

        if eval_result.cm is not None:
            cm_label = "" if eval_result.cm.sum() > 0 else "empty"
            fig = plot_confusion_matrix(eval_result.cm, class_names, label=cm_label)
            img = figure_to_numpy(fig)
            try:
                vis.add_image(f"{tag_prefix}confusion_matrix", img, step=step)
            except Exception:
                pass
            import matplotlib.pyplot as plt

            plt.close(fig)

        for lbl, rcm in eval_result.range_cms.items():
            if rcm is None:
                continue
            cm_label = lbl if rcm.sum() > 0 else f"{lbl} (empty)"
            fig = plot_confusion_matrix(rcm, class_names, label=cm_label)
            img = figure_to_numpy(fig)
            tag = f"confusion_matrix_{lbl.replace('-', '_').replace(' ', '_')}"
            try:
                vis.add_image(f"{tag_prefix}{tag}", img, step=step)
            except Exception:
                pass
            import matplotlib.pyplot as plt

            plt.close(fig)
