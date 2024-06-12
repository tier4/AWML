import json
import os
from typing import Any, Dict, List, Tuple

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.evaluate import DetectionEval


class T4DetectionConfig(DetectionConfig):
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
        max_boxes_per_sample: int,
        mean_ap_weight: int,
    ):
        assert class_range.keys() == set(
            class_names), "class_range must have keys for all classes."
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


class T4DetectionEvaluation(DetectionEval):

    def __init__(
        self,
        config: T4DetectionConfig,
        result_path: str,
        scene: str,
        output_dir: str,
        ground_truth_boxes: EvalBoxes,
        prediction_boxes: EvalBoxes,
        verbose: bool = True,
    ):
        """
        Initialize a DetectionEval object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param scene: The scene token to evaluate on
        :param output_dir: Folder to save plots and results to.
        :param ground_truth_boxes: The ground truth boxes.
        :param prediction_boxes: The predicted boxes.
        :param verbose: Whether to print to stdout.
        """
        self.result_path = result_path
        self.scene = scene
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.gt_boxes = ground_truth_boxes
        self.pred_boxes = prediction_boxes

        # Check result file exists.
        assert os.path.exists(
            result_path), "Error: The result file does not exist!"

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print("Initializing nuScenes detection evaluation")
        assert set(self.pred_boxes.sample_tokens) == set(
            self.gt_boxes.sample_tokens
        ), "Samples in split doesn't match samples in predictions."

        self.sample_tokens = self.gt_boxes.sample_tokens

    def run_and_save_eval(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Main function that loads the evaluation code, visualizes samples,
        runs the evaluation and renders stat plots.

        :return: A dict that stores the high-level metrics and meta data.
        """
        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Dump the metric data, meta and metrics to disk.
        print("Saving metrics to: %s" % self.output_dir)
        metrics_summary = metrics.serialize()
        # metrics_summary["meta"] = self.meta.copy()
        with open(os.path.join(self.output_dir, "metrics_summary.json"),
                  "w") as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, "metrics_details.json"),
                  "w") as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        mean_AP = "{:.3g}".format(metrics_summary["mean_ap"])
        header = [
            "class_name",
            "mAP",
            "AP@0.5m",
            "AP@1.0m",
            "AP@2.0m",
            "AP@4.0m",
            #"error@trans_err",
            #"error@scale_err",
            #"error@orient_err",
            #"error@vel_err",
            #"error@attr_err",
        ]
        data = []
        class_aps = metrics_summary["mean_dist_aps"]
        class_tps = metrics_summary["label_tp_errors"]
        label_aps = metrics_summary["label_aps"]
        for class_name in class_aps.keys():
            data.append([
                class_name,
                "{:.1f}".format(class_aps[class_name] * 100.0),
                "{:.1f}".format(label_aps[class_name][0.5] * 100.0),
                "{:.1f}".format(label_aps[class_name][1.0] * 100.0),
                "{:.1f}".format(label_aps[class_name][2.0] * 100.0),
                "{:.1f}".format(label_aps[class_name][4.0] * 100.0),
                #"{:.3g}".format(class_tps[class_name]["trans_err"]),
                #"{:.3g}".format(class_tps[class_name]["scale_err"]),
                #"{:.3g}".format(class_tps[class_name]["orient_err"]),
                #"{:.3g}".format(class_tps[class_name]["vel_err"]),
                #"{:.3g}".format(class_tps[class_name]["attr_err"]),
            ])
        metrics_table = {
            "header": header,
            "data": data,
            "total_mAP": mean_AP,
        }
        return metrics_summary, metrics_table
