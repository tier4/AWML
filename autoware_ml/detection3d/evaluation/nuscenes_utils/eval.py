import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from nuscenes.eval.common.data_classes import EvalBox, EvalBoxes
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import DetectionEval as _DetectionEval


def toEvalBoxes(nusc_boxes: Dict[str, List[Dict]], box_cls: EvalBox = DetectionBox) -> EvalBoxes:
    """

    nusc_boxes = {
        "sample_token_1": [
            {
                "sample_token": str,
                "translation": List[float], (x, y, z)
                "size": List[float], (width, length, height)
                "rotation": List[float], (w, x, y, z)
                "velocity": List[float], (vx, vy)
                "detection_name": str,
                "detection_score": float,
                "attribute_name": str,
            },
            ...
        ],
        ...
    }

    Args:
        nusc_boxes (Dict[List[Dict]]): [description]
        box_cls (EvalBox, optional): [description]. Defaults to DetectionBox.

    Returns:
        EvalBoxes: [description]
    """
    return EvalBoxes.deserialize(nusc_boxes, box_cls)

class DetectionConfig:
    """Data class that specifies the detection evaluation settings."""

    def __init__(
        self,
        class_names: List[str],
        class_range: Dict[str, int],
        dist_fcn: str,
        dist_ths: List[float],
        dist_th_tp: float,
        min_recall: float,
        min_precision: float,
        max_boxes_per_sample: float,
        mean_ap_weight: int,
    ):
        # assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = class_names

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """Serialize instance into json-friendly format."""
        return {
            "class_names": self.class_names,
            "class_range": self.class_range,
            "dist_fcn": self.dist_fcn,
            "dist_ths": self.dist_ths,
            "dist_th_tp": self.dist_th_tp,
            "min_recall": self.min_recall,
            "min_precision": self.min_precision,
            "max_boxes_per_sample": self.max_boxes_per_sample,
            "mean_ap_weight": self.mean_ap_weight,
        }

    @classmethod
    def deserialize(cls, content: dict):
        """Initialize from serialized dictionary."""
        return cls(
            content["class_names"],
            content["class_range"],
            content["dist_fcn"],
            content["dist_ths"],
            content["dist_th_tp"],
            content["min_recall"],
            content["min_precision"],
            content["max_boxes_per_sample"],
            content["mean_ap_weight"],
        )

    @property
    def dist_fcn_callable(self):
        """Return the distance function corresponding to the dist_fcn string."""
        if self.dist_fcn == "center_distance":
            return center_distance
        else:
            raise Exception("Error: Unknown distance function %s!" % self.dist_fcn)


class nuScenesDetectionEval(_DetectionEval):
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.
    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.
    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.
    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.
    Please see https://www.nuscenes.org/object-detection for more details.
    """

    def __init__(
        self,
        config: DetectionConfig,
        result_boxes: Dict,
        gt_boxes: Dict,
        meta: Dict,
        eval_set: str,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize a DetectionEval object.
        :param config: A DetectionConfig object.
        :param result_boxes: result bounding boxes.
        :param gt_boxes: ground-truth bounding boxes.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.cfg = config
        self.meta = meta
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.pred_boxes: EvalBoxes = toEvalBoxes(result_boxes)
        self.gt_boxes: EvalBoxes = toEvalBoxes(gt_boxes)

        assert set(self.pred_boxes.sample_tokens) == set(
            self.gt_boxes.sample_tokens
        ), "Samples in split doesn't match samples in predictions."

        self.sample_tokens = self.gt_boxes.sample_tokens
