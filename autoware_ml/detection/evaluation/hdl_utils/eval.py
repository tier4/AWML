"""Evaluation utilities powered by Human Dataware Lab. Co., Ltd."""

import json
import logging
from typing import Dict, List, Optional, Tuple

from mmdet3d.structures import limit_period
from mmdet3d.structures.ops.iou3d_calculator import bbox_overlaps_nearest_3d
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch

from .utils import (
    EvalLiDARInstance3DBoxes,
    MultiDistanceMatrix,
    _get_attributes,
    _mmdet3d_bbox_from_dicts,
)


class InvalidHDLEvaluationConfigError(Exception):
    """Error for the case HDLEvaluationConfig is invalid"""
    pass


class HDLEvaluationConfig:
    """Config for HDLEvaluator."""
    es_box_filter: str = ""  # filter for the estimated boxes
    gt_box_filter: str = ""  # filter for the ground-truth boxes

    # Condition of determining that the boxes are overlapped.
    # Choose one of the followings:
    #   - iou: 2D (BEV) IoU of the boxes are above the threshold given as a float value.
    #   - distance: 3D center-distance of the boxes are below the threshold given as a float value.
    #   - iou-or-distance: either 2D (BEV) IoU of the boxes are above the threshold or
    #                      3D center-distance of the boxes are below the threshold.
    #                      In this case, the threshold value must be given as a dict value
    #                      with 'iou' and 'distance' as keys.
    overlap_condition = dict(criteria="iou", threshold=0.5)

    hard_decision_threshold: Optional[float] = None  # threshold for hard decision
    f1_value_beta: float = 1.0  # weight for f1 value

    def __init__(self, **options):
        """Initialize HDLEvaluationConfig.

        Args:
            **es_box_filter (str): A Pandas DataFrame query string to filter the estimated boxes.
            **gt_box_filter (str): A Pandas DataFrame query string to filter the ground-truth.
            **overlap_condition (dict): Define the condition of True-Positives
                                        (GT and ES boxes are overlapped).

        """
        super(HDLEvaluationConfig, self).__init__()

        # Parse options
        options_not_parsed = list(options.keys())
        if options is not None:
            for attr in [content[0] for content in _get_attributes(self)]:
                if attr in options.keys():
                    options_not_parsed.remove(attr)
                    setattr(self, attr, options[attr])

        # Assertion
        if len(options_not_parsed) > 0:
            raise InvalidHDLEvaluationConfigError(f"unsupported arguments: {options_not_parsed}")

        # Validation
        try:
            self.validate()
        except AssertionError as e:
            raise InvalidHDLEvaluationConfigError(str(e))

    def __eq__(self, other):
        """Compare self with the other."""
        return json.dumps(self.serialize()) == json.dumps(other.serialize())

    def validate(self):
        """Validate configurations."""
        assert isinstance(self.es_box_filter, str)
        assert isinstance(self.gt_box_filter, str)
        assert len(set(self.overlap_condition.keys()).difference(["criteria", "threshold"])) == 0
        assert self.overlap_condition["criteria"] in ["iou", "distance", "iou-or-distance"]
        if self.overlap_condition["criteria"] in ["iou", "distance"]:
            assert isinstance(self.overlap_condition["threshold"], float)
        elif self.overlap_condition["criteria"] in ["iou-or-distance"]:
            assert isinstance(self.overlap_condition["threshold"], dict)
            assert "iou" in self.overlap_condition["threshold"].keys()
            assert "distance" in self.overlap_condition["threshold"].keys()
            assert (
                0.0 < self.overlap_condition["threshold"]["iou"] < 1.0
            ), "IoU threshold must be >0.0 and <1.0"
            assert (
                self.overlap_condition["threshold"]["distance"] > 0.0
            ), "Distance threshold must be >0.0"

    def serialize(self):
        """Serialize instance into json-friendly format."""
        options = dict()
        for attr in sorted([content[0] for content in _get_attributes(self)]):
            options.update({attr: getattr(self, attr)})
        return options

    @classmethod
    def deserialize(cls, content: dict):
        """Initialize from serialized dictionary."""
        return HDLEvaluationConfig(**content)


class HDLEvaluator:
    """An evaluator for object detection powered by Human Dataware Lab."""

    summarizable_criterion = [
        "num-es-boxes",
        "num-gt-boxes",
        "TPR",
        "FNR",
        "FPR",
        "precision",
        "recall",
        "f-value",
        "AP",
        "best-f-value",
        "best-f-value-precision",
        "best-f-value-recall",
        "best-f-value-threshold",
        "APH",
        "best-f-value-h",
        "best-f-value-precision-h",
        "best-f-value-recall-h",
        "best-f-value-threshold-h",
    ]

    def __init__(
        self,
        config: HDLEvaluationConfig,
        es_boxes: dict,
        gt_boxes: dict,
    ):
        """Initialize HDLEvaluator.

        Args:
            config (HDLEvaluationConfig): Evaluation condition.
            es_boxes (dict): Estimated bounding boxes loaded with `mmengine.load`.
            gt_boxes (dict): Ground-truth bounding boxes loaded with `mmengine.load`.

        """
        super(HDLEvaluator, self).__init__()
        self.logger = logging.getLogger(name=type(self).__name__)
        self.config = config
        self.config.validate()

        # Store boxes as DataFrame
        self.es_boxes_df = {key: pd.DataFrame.from_dict(value) for key, value in es_boxes.items()}
        self.gt_boxes_df = {key: pd.DataFrame.from_dict(value) for key, value in gt_boxes.items()}

        # Apply filters and convert bboxes to EvalLiDARInstance3DBoxes
        self._es_boxes, self._gt_boxes = self._prepare_bboxes()

        self._gt_true_positives = []
        self._gt_false_negatives = []
        self._es_false_positives = []
        self._es_scores = []

    @property
    def es_boxes(self) -> Dict[str, EvalLiDARInstance3DBoxes]:
        """Filtered es_boxes"""
        return self._es_boxes

    @property
    def gt_boxes(self) -> Dict[str, EvalLiDARInstance3DBoxes]:
        """Filtered gt_boxes"""
        return self._gt_boxes

    @property
    def es_scores(self) -> List[float]:
        """Detection scores."""
        return self._es_scores

    @property
    def es_true_positives(self) -> List[bool]:
        """boolean of estimated bounding boxes (length: number of detections)."""
        return list(map(lambda b: not b, self._es_false_positives))

    @property
    def num_es_true_positives(self) -> int:
        """Number of trues in self.es_true_positives."""
        if len(self.es_true_positives) == 0:
            return np.nan
        else:
            return self.es_true_positives.count(True)

    @property
    def es_false_positives(self) -> List[bool]:
        """boolean of false positives (length: number of detections)."""
        return self._es_false_positives

    @property
    def num_es_false_positives(self) -> int:
        """Number of trues in self.es_false_positives."""
        if len(self.es_false_positives) == 0:
            return np.nan
        else:
            return self.es_false_positives.count(True)

    @property
    def gt_true_positives(self) -> List[bool]:
        """boolean of ground-truth bounding boxes (length: number of labels)."""
        return self._gt_true_positives

    @property
    def num_gt_true_positives(self) -> int:
        """Number of trues in self.gt_true_positives."""
        if len(self.gt_true_positives) == 0:
            return np.nan
        else:
            return self.gt_true_positives.count(True)

    @property
    def gt_false_negatives(self) -> List[bool]:
        """boolean of false negatives (length: number of labels)."""
        return self._gt_false_negatives

    @property
    def num_gt_false_negatives(self) -> int:
        """Number of trues in self.gt_false_negatives."""
        if len(self.gt_false_negatives) == 0:
            return np.nan
        else:
            return self.gt_false_negatives.count(True)

    @property
    def _cond_fn(self) -> callable:
        if self.config.overlap_condition["criteria"] == "iou":

            def cond_fn(value):
                return np.greater_equal(value, self.config.overlap_condition["threshold"])

        elif self.config.overlap_condition["criteria"] == "distance":

            def cond_fn(value):
                return np.less_equal(value, self.config.overlap_condition["threshold"])

        elif self.config.overlap_condition["criteria"] == "iou-or-distance":

            def cond_fn(value):
                threshold = self.config.overlap_condition["threshold"]
                return np.logical_or(
                    np.greater_equal(value["iou"], threshold["iou"]),
                    np.less_equal(value["distance"], threshold["distance"]),
                )

        else:
            raise InvalidHDLEvaluationConfigError(
                f"cond_fn is not implemented for {self.config.overlap_condition['criteria']}"
            )

        return cond_fn

    @property
    def _minmax_fn(self) -> callable:
        if self.config.overlap_condition["criteria"] == "iou":
            minmax_fn = np.nanmax
        elif self.config.overlap_condition["criteria"] == "distance":
            minmax_fn = np.nanmin
        elif self.config.overlap_condition["criteria"] == "iou-or-distance":

            def minmax_fn(matrix, axis=None):
                return {
                    "iou": np.nanmax(matrix["iou"], axis=axis),
                    "distance": np.nanmin(matrix["distance"], axis=axis),
                }

        else:
            raise InvalidHDLEvaluationConfigError(
                f"minmax_fn is not implemented for {self.config.overlap_condition['criteria']}"
            )

        return minmax_fn

    @property
    def _argminmax_fn(self) -> callable:
        if self.config.overlap_condition["criteria"] == "iou":
            argminmax_fn = np.nanargmax
        elif self.config.overlap_condition["criteria"] == "distance":
            argminmax_fn = np.nanargmin
        elif self.config.overlap_condition["criteria"] == "iou-or-distance":

            def argminmax_fn(matrix, axis=None):
                threshold = self.config.overlap_condition["threshold"]
                minmax: Dict[str, callable] = self._minmax_fn(matrix, axis=axis)
                argminmax_iou = np.nanargmax(matrix["iou"], axis=axis)
                argminmax_distance = np.nanargmin(matrix["distance"], axis=axis)
                if axis is None:
                    minmax = {key: np.array([value]) for key, value in minmax.items()}
                    argminmax_iou = np.array([argminmax_iou])
                    argminmax_distance = np.array([argminmax_distance])

                assert len(argminmax_iou) == len(argminmax_distance)
                argminmax = []
                for idx in range(len(argminmax_iou)):
                    if argminmax_iou[idx] == argminmax_distance[idx]:
                        argminmax.append(argminmax_iou[idx])
                    else:
                        if np.greater_equal(
                            minmax["iou"][idx], threshold["iou"]
                        ) and np.less_equal(minmax["distance"][idx], threshold["distance"]):
                            # In case that both IoU and center-distance satisfy overlapping condition,
                            # we firstly calculate the ratio of `oracle best to the value` out of
                            # `oracle best to the threshold` in both IoU and center-distance.
                            # Then we take the one whose ratio is lower than the other one.
                            # In other words, we choose the one which is relatively close to the oracle best
                            # and reject the other one that is relatively close to the threshold value.
                            ratio_iou = abs((1.0 - minmax["iou"][idx]) / (1.0 - threshold["iou"]))
                            ratio_distance = abs(
                                (0.0 - minmax["distance"][idx]) / (0.0 - threshold["distance"])
                            )
                            if ratio_iou < ratio_distance:
                                argminmax.append(argminmax_iou[idx])
                            else:
                                argminmax.append(argminmax_distance[idx])
                        elif np.greater_equal(minmax["iou"][idx], threshold["iou"]):
                            argminmax.append(argminmax_iou[idx])
                        elif np.less_equal(minmax["distance"][idx], threshold["distance"]):
                            argminmax.append(argminmax_distance[idx])
                        else:
                            # If no objects are overlapped based on neither IoU nor center-distance criterion,
                            # then return 0 according to `np.nanargmin` specification.
                            argminmax.append(0)
                argminmax = np.array(argminmax)
                assert argminmax.shape == argminmax_iou.shape == argminmax_distance.shape

                return argminmax

        else:
            raise InvalidHDLEvaluationConfigError(
                f"argminmax_fn is not implemented for {self.config.overlap_condition['criteria']}"
            )

        return argminmax_fn

    def evaluate(self):
        """Conduct evaluation."""
        scores = self._evaluate(
            es_boxes=list(self.es_boxes.values()),
            gt_boxes=list(self.gt_boxes.values()),
        )
        scores["evaluation_conditions"] = self.config.serialize()
        return scores

    def get_confusion_matrix(self, labels: List[str]):
        """Get confusion matrix.
        Firstly, this function finds detected boxes that overlapped to any of the GT labels.
        Then, a confusion matrix is calculated based on the GT labels and the overlapped boxes.
        The condition of overlapping is defined by HDLEvaluationConfig and if multiple boxes are overlapped to
        the same GT label, the one with the highest score is selected.

        Args:
            labels (list[str]): A list of a class label.

        Returns:
            (pandas.DataFrame): Confusion matrix.

        """
        _cm = self._get_confusion_matrix(
            es_boxes=list(self.es_boxes.values()),
            gt_boxes=list(self.gt_boxes.values()),
            labels=labels,
        )
        if _cm.size > 0:
            cm = pd.DataFrame(
                _cm,
                index=[f"{label} (Ground-truth)" for label in labels],
                columns=[f"{label} (Predicted)" for label in labels],
            )
        else:
            cm = pd.DataFrame()
        return cm

    def _prepare_bboxes(self):
        """Prepare bounding boxes."""
        # Filter boxes
        es_boxes_df_filtered = {
            key: value.query(self.config.es_box_filter)
            if len(value) > 0 and self.config.es_box_filter != ""
            else value
            for key, value in self.es_boxes_df.items()
        }
        gt_boxes_df_filtered = {
            key: value.query(self.config.gt_box_filter)
            if len(value) > 0 and self.config.gt_box_filter != ""
            else value
            for key, value in self.gt_boxes_df.items()
        }

        # Convert DataFrames to BBox objects
        es_boxes = {
            key: _mmdet3d_bbox_from_dicts(value.to_dict(orient="records"))
            for key, value in es_boxes_df_filtered.items()
        }
        gt_boxes = {
            key: _mmdet3d_bbox_from_dicts(value.to_dict(orient="records"))
            for key, value in gt_boxes_df_filtered.items()
        }

        return es_boxes, gt_boxes

    def _get_overlap_distance_matrix(
        self, bboxes1: EvalLiDARInstance3DBoxes, bboxes2: EvalLiDARInstance3DBoxes
    ):
        """Get a matrix of box distances.

        Args:
            bboxes1 (EvalLiDARInstance3DBoxes): Bounding boxes to investigate (length: N).
            bboxes2 (EvalLiDARInstance3DBoxes): Bounding boxes to refer (length: M).

        Returns:
            (ndarray): A matrix with shape (N, M) containing the distance of boxes corresponding to each row and col.

        """
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.empty(shape=(len(bboxes1), len(bboxes2)))
        if self.config.overlap_condition["criteria"] == "iou":
            ious = bbox_overlaps_nearest_3d(
                bboxes1.tensor, bboxes2.tensor, mode="iou", is_aligned=False, coordinate="lidar"
            ).numpy()
            return ious
        elif self.config.overlap_condition["criteria"] == "distance":
            distances = []
            for idx in range(len(bboxes1)):
                dists = np.sqrt(
                    ((bboxes2.center.numpy() - bboxes1.center[idx].numpy()) ** 2).sum(axis=1)
                )
                distances.append(dists)
            distances = np.array(distances)
            return distances
        elif self.config.overlap_condition["criteria"] == "iou-or-distance":
            distances = []
            for idx in range(len(bboxes1)):
                dists = np.sqrt(
                    ((bboxes2.center.numpy() - bboxes1.center[idx].numpy()) ** 2).sum(axis=1)
                )
                distances.append(dists)
            distances = np.array(distances)
            ious = bbox_overlaps_nearest_3d(
                bboxes1.tensor, bboxes2.tensor, mode="iou", is_aligned=False, coordinate="lidar"
            ).numpy()
            return MultiDistanceMatrix({"iou": ious, "distance": distances})
        else:
            raise InvalidHDLEvaluationConfigError(
                f"unsupported overlap condition: {self.config.overlap_condition}"
            )

    def _has_overlapped_boxes(
        self, bboxes1: EvalLiDARInstance3DBoxes, bboxes2: EvalLiDARInstance3DBoxes
    ):
        """Investigate if boxes in boxes1 have overlapped boxes in bbox2.

        Args:
            bboxes1 (EvalLiDARInstance3DBoxes): Bounding boxes to investigate (length: N).
            bboxes2 (EvalLiDARInstance3DBoxes): Bounding boxes to refer (length: M).

        Returns:
            (ndarray): array of boolean indicating overlaps with shape (N,)

        """
        if len(bboxes1) == 0:
            return np.empty(shape=(0,))
        if len(bboxes2) == 0:
            return np.array([False] * len(bboxes1))

        distance_matrix = self._get_overlap_distance_matrix(bboxes1, bboxes2)

        if self.config.overlap_condition["criteria"] == "iou":
            return np.max(distance_matrix > self.config.overlap_condition["threshold"], axis=1)
        elif self.config.overlap_condition["criteria"] == "distance":
            return np.max(distance_matrix < self.config.overlap_condition["threshold"], axis=1)
        elif self.config.overlap_condition["criteria"] == "iou-or-distance":
            return np.logical_or(
                np.max(
                    distance_matrix["iou"] > self.config.overlap_condition["threshold"]["iou"],
                    axis=1,
                ),
                np.max(
                    distance_matrix["distance"]
                    < self.config.overlap_condition["threshold"]["distance"],
                    axis=1,
                ),
            )
        else:
            raise InvalidHDLEvaluationConfigError(
                f"unsupported overlap condition: {self.config.overlap_condition}"
            )

    def _calculate_tp_fp_and_fn(
        self, gt_boxes: EvalLiDARInstance3DBoxes, es_boxes: EvalLiDARInstance3DBoxes
    ):
        """Calculate TP, FP, and FN.

        Args:
            gt_boxes (EvalLiDARInstance3DBoxes): Ground-truth boxes (length N).
            es_boxes (EvalLiDARInstance3DBoxes): Detected boxes (length M).

        Returns:
            list[bool]: A list of boolean indicating TP in gt_boxes (length N).
            list[bool]: A list of boolean indicating FP in es_boxes (length M).
            list[bool]: A list of boolean indicating FN in gt_boxes (length N).

        """
        if len(gt_boxes) == 0:
            return [], [True for _ in range(len(es_boxes))], []
        if len(es_boxes) == 0:
            return [False for _ in range(len(gt_boxes))], [], [True for _ in range(len(gt_boxes))]

        distance_matrix = self._get_overlap_distance_matrix(gt_boxes, es_boxes)

        gt_tp_flags = np.zeros(len(gt_boxes), dtype=np.bool)
        es_tp_flags = np.zeros(len(es_boxes), dtype=np.bool)
        while self._cond_fn(self._minmax_fn(distance_matrix)):
            row, col = np.unravel_index(
                self._argminmax_fn(distance_matrix), shape=distance_matrix.shape
            )
            gt_tp_flags[row] = True
            es_tp_flags[col] = True
            distance_matrix[row, :] = np.nan
            distance_matrix[:, col] = np.nan
        tp = gt_tp_flags.tolist()
        fp = np.logical_not(es_tp_flags).tolist()
        fn = np.logical_not(gt_tp_flags).tolist()

        return tp, fp, fn

    def _calculate_ap_and_best_f_value(
        self,
        hard_decision_threshold=None,
        use_metrics_weighted_by_heading=False,
        beta=1.0,
    ):
        """Calculate Average Precision and the best f-value.

        Calculation of AP refers nuscenes-devkit:
        https://github.com/nutonomy/nuscenes-devkit/blob/87b88fe37ad503e6e35dc2546ae1bf74f4ef6c01/python-sdk/nuscenes/eval/detection/algo.py

        Returns:
            (tuple): A tuple of AP, the best f-value, a threshold value of scores that gives the best f-value.

        """
        if use_metrics_weighted_by_heading:
            es_boxes = list(self.es_boxes.values())
            gt_boxes = list(self.gt_boxes.values())
            tph = []
            for esb, gtb in zip(es_boxes, gt_boxes):
                if len(esb.tensor) == 0:
                    tph_ = np.empty(shape=(0,))
                elif len(gtb.tensor) == 0:
                    tph_ = np.array([0.0] * len(esb.tensor))
                else:
                    distance_matrix = self._get_overlap_distance_matrix(esb, gtb)
                    tp_ = np.max(self._cond_fn(distance_matrix), axis=1)
                    es_yaw_tp_ = esb.tensor.numpy()[tp_, -1]
                    gt_yaw_tp_ = gtb.tensor.numpy()[self._argminmax_fn(distance_matrix, axis=1)][
                        tp_, -1
                    ]
                    assert len(es_yaw_tp_) == len(gt_yaw_tp_)
                    tph_ = tp_.astype(np.float32)
                    # consider periodic value
                    yaw_diff = limit_period(torch.tensor(np.abs(es_yaw_tp_ - gt_yaw_tp_))).numpy()
                    # NOTE(kan-bayashi): np.abs is added to accept 180 degree misprediction
                    tph_[tp_] = np.abs(1 - yaw_diff / (np.pi / 2))
                tph += [tph_]
            # Use TP weighted by heading as TP
            tp = np.concatenate(tph)
            fp = tp == 0
        else:
            tp = np.array(self.es_true_positives)
            fp = np.array(self.es_false_positives)
        scores = np.array(self.es_scores)
        npos = sum([len(labels) for labels in list(self.gt_boxes.values())])
        min_precision = 0.1
        min_recall = 0.1

        assert len(tp) == len(fp) == len(scores)
        if len(scores) == 0 or npos == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        # Sort tp and fp
        desc_score_indices = np.argsort(scores)[::-1]
        tp = tp[desc_score_indices]
        fp = fp[desc_score_indices]
        scores = scores[desc_score_indices]

        # Accumulate.
        tp = np.cumsum(tp).astype(float)
        fp = np.cumsum(fp).astype(float)

        # Calculate precisions and recalls and f-values.
        prec = tp / (fp + tp)
        rec = tp / float(npos)
        # NOTE(kan-bayashi): Use weighted f1 score
        f_values = np.array(
            [
                (1 + beta**2) * p * r / (beta**2 * p + r) if p + r != 0.0 else 0.0
                for p, r in zip(prec, rec)
            ]
        )

        if hard_decision_threshold is None:
            best_f_value_idx = np.argmax(f_values)
            best_f_value_threshold = float(scores[best_f_value_idx])
            best_f_value_precision = float(prec[best_f_value_idx])
            best_f_value_recall = float(rec[best_f_value_idx])
            best_f_value = float(f_values[best_f_value_idx])
        else:
            if scores[0] < hard_decision_threshold:
                # Detection scores are below hard_decision_threshold
                best_f_value_threshold = None
                best_f_value_precision = None
                best_f_value_recall = None
                best_f_value = None
            else:
                best_f_value_idx = np.where(scores >= hard_decision_threshold)[0][-1]
                best_f_value_threshold = hard_decision_threshold
                best_f_value_precision = float(prec[best_f_value_idx])
                best_f_value_recall = float(rec[best_f_value_idx])
                best_f_value = float(f_values[best_f_value_idx])

        # Calculate AP
        rec_interp = np.linspace(0, 1, 101)
        prec = np.interp(rec_interp, rec, prec, right=0)
        prec = prec[
            round(100 * min_recall) + 1 :
        ]  # Clip low recalls. +1 to exclude the min recall bin.
        prec -= min_precision  # Clip low precision
        prec[prec < 0] = 0
        ap = float(np.mean(prec)) / (1.0 - min_precision)

        return (
            ap,
            best_f_value,
            best_f_value_precision,
            best_f_value_recall,
            best_f_value_threshold,
        )

    def _get_confusion_matrix(
        self,
        es_boxes: List[EvalLiDARInstance3DBoxes],
        gt_boxes: List[EvalLiDARInstance3DBoxes],
        labels: List[str],
    ) -> Tuple[np.array, List[str]]:
        """Get confusion matrix.
        This function finds detected boxes overlapped to GT boxes and calculate a confusion matrix.
        When multiple detections are overlapped to a single GT box,
        the one with the highest score is used.

        Args:
            es_boxes (list): A list of EvalLiDARInstance3DBoxes.
            gt_boxes (list): A list of EvalLiDARInstance3DBoxes.
            labels (list[str]): A list of a class label.

        Returns:
            (ndarray): ndarray of shape (n_classes, n_classes).
                Confusion matrix whose i-th row and j-th column entry
                indicates the number of samples with true label being i-th
                class and predicted label being j-th class.
            (list): List of labels which corresponds to the columns and rows
                of the confusion matrix.

        """
        y_true = []
        y_pred = []
        for frame_idx in range(len(gt_boxes)):
            for obj_idx in range(len(gt_boxes[frame_idx])):
                gt_box = gt_boxes[frame_idx][obj_idx : obj_idx + 1]
                gt_name = gt_boxes[frame_idx].names[obj_idx]

                # Find an overlapped box with the highest score in es_boxes[frame_idx]
                overlapping_flags = self._has_overlapped_boxes(es_boxes[frame_idx], gt_box)
                if len(overlapping_flags) == 0 or not np.max(overlapping_flags):
                    continue  # Skip if no detections are overlapped to the GT box
                _es_scores = es_boxes[frame_idx].score.cpu().clone().numpy()
                _es_scores[np.logical_not(overlapping_flags)] = -1.0
                overlapped_box_idx = int(np.argmax(_es_scores))
                es_name = es_boxes[frame_idx].names[overlapped_box_idx]

                # Substitute
                y_true.append(gt_name)
                y_pred.append(es_name)

        if len(y_true) > 0 and len(y_pred) > 0 and len(labels) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=labels)
        else:
            cm = np.array([[]])

        return cm

    def _evaluate(
        self, es_boxes: List[EvalLiDARInstance3DBoxes], gt_boxes: List[EvalLiDARInstance3DBoxes]
    ):
        """Conduct evaluation.

        Args:
            es_boxes (list): A list of EvalLiDARInstance3DBoxes.
            gt_boxes (list): A list of EvalLiDARInstance3DBoxes.

        Returns:
            (dict): Evaluation results.

        """
        # Check
        assert len(gt_boxes) == len(es_boxes)
        if len(gt_boxes) == 0:
            raise EOFError()

        # Analyze
        num_gt_boxes = sum([len(labels) for labels in gt_boxes])
        num_es_boxes = sum([len(labels) for labels in es_boxes])

        # TP, FP, FN
        _tp = []
        _fp = []
        _fn = []
        for frame_gt_boxes, frame_es_boxes in zip(gt_boxes, es_boxes):
            tp, fp, fn = self._calculate_tp_fp_and_fn(frame_gt_boxes, frame_es_boxes)
            _tp += tp
            _fp += fp
            _fn += fn
        true_positives = _tp.count(True)
        false_positives = _fp.count(True)
        false_negatives = _fn.count(True)
        self._gt_true_positives = _tp
        self._es_false_positives = _fp
        self._gt_false_negatives = _fn
        assert true_positives + false_negatives == num_gt_boxes
        assert true_positives + false_positives == num_es_boxes

        # TPR, FPR, FNR
        true_positive_rate = (
            float(true_positives) / float(num_gt_boxes) if num_gt_boxes != 0 else np.nan
        )
        false_positive_rate = (
            float(false_positives) / float(num_es_boxes) if num_es_boxes != 0 else np.nan
        )
        false_negative_rate = (
            float(false_negatives) / float(num_gt_boxes) if num_gt_boxes != 0 else np.nan
        )

        # Precision
        precision = (
            float(true_positives) / float(true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else np.nan
        )

        # Recall
        recall = (
            float(true_positives) / float(true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else np.nan
        )

        # F-value
        f_value = (
            2.0 * recall * precision / (recall + precision)
            if (recall + precision) > 0.0
            else np.nan
        )

        # AP and best-F
        _es_scores = []
        for ess in es_boxes:
            _es_scores += list(ess.score.cpu().numpy())
        self._es_scores = _es_scores
        (
            ap,
            best_f_value,
            best_f_value_precision,
            best_f_value_recall,
            best_f_value_threshold,
        ) = self._calculate_ap_and_best_f_value(
            hard_decision_threshold=self.config.hard_decision_threshold,
            use_metrics_weighted_by_heading=False,
            beta=self.config.f1_value_beta,
        )
        (
            aph,
            best_f_value_h,
            best_f_value_precision_h,
            best_f_value_recall_h,
            best_f_value_threshold_h,
        ) = self._calculate_ap_and_best_f_value(
            hard_decision_threshold=self.config.hard_decision_threshold,
            use_metrics_weighted_by_heading=True,
            beta=self.config.f1_value_beta,
        )

        result = {
            "num-gt-boxes": num_gt_boxes,
            "num-es-boxes": num_es_boxes,
            "TP": true_positives,
            "TPR": true_positive_rate,
            "FN": false_negatives,
            "FNR": false_negative_rate,
            "FP": false_positives,
            "FPR": false_positive_rate,
            "precision": precision,
            "recall": recall,
            "f-value": f_value,
            "AP": ap,
            "best-f-value": best_f_value,
            "best-f-value-precision": best_f_value_precision,
            "best-f-value-recall": best_f_value_recall,
            "best-f-value-threshold": best_f_value_threshold,
            "APH": aph,
            "best-f-value-h": best_f_value_h,
            "best-f-value-precision-h": best_f_value_precision_h,
            "best-f-value-recall-h": best_f_value_recall_h,
            "best-f-value-threshold-h": best_f_value_threshold_h,
        }
        return result
