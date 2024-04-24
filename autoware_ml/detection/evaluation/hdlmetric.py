from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from awml_det3d.registry import METRICS, TRANSFORMS
from mmengine import ConfigDict, load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from numpy import who

from autoware_ml.detection.evaluation.hdl_utils import (HDLEvaluationConfig,
                                                        HDLEvaluator)
from autoware_ml.detection.evaluation.utils.eval import (
    print_metrics_table, validate_model_output_mapping)
from autoware_ml.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes


@METRICS.register_module()
class HDLMetric(BaseMetric):
    """Metric utilising the Human Dataware Lab evaluator"""

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        name_mapping: Optional[ConfigDict] = None,  # type should be derived from BaseNameMapping
        model_mapping: Optional[dict] = None,
        data_mapping: Optional[dict] = None,
        hdl_pedestrian_matching: Optional[float] = None,
        class_names: List[str] = [],
        backend_args: Optional[dict] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        with_velocity: bool = False,
        overlap_criterion: str = "iou",
        overlap_threshold: float = 0.5,
        hard_decision_threshold: Optional[float] = None,
        f1_value_beta: float = 1.0,
        dataset_origin: Tuple[float, float, float] = (0.5, 0.5, 0.0),
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        if name_mapping:
            self.name_mapping = TRANSFORMS.build(name_mapping)
        else:
            self.name_mapping = None
        self.class_names = class_names
        self.model_mapping = model_mapping  # maps outputs from the model
        self.data_mapping = data_mapping  # maps gts
        if self.data_mapping is not None:
            self.class_names = [self.data_mapping.get(name, name) for name in self.class_names]
        if self.model_mapping is not None:
            validate_model_output_mapping(self.model_mapping, self.class_names)
        self.ann_file = ann_file
        self.backend_args = backend_args or {}
        self.data_infos = load(self.ann_file, backend_args=self.backend_args)["data_list"]
        self.hdl_pedestrian_matching = hdl_pedestrian_matching
        self.with_velocity = with_velocity
        self.dataset_origin = dataset_origin

        # HDL evaluator config parameters
        self.overlap_condition = {
            "criteria": overlap_criterion,
            "threshold": overlap_threshold,
        }
        self.hard_decision_threshold = hard_decision_threshold
        self.f1_value_beta = f1_value_beta

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        pred_boxes = {key: value["pred_instances_3d"] for key, value in enumerate(results)}
        gt_boxes = {key: value["instances"] for key, value in enumerate(self.data_infos)}

        for key in pred_boxes.keys():
            detection_names = self._map_labels_to_names(pred_boxes[key]["labels_3d"])
            if self.model_mapping:
                detection_names = [self.model_mapping.get(name, name) for name in detection_names]
            pred_boxes[key]["detection_names"] = detection_names

        gt_boxes = self._prepare_gt_boxes(gt_boxes)

        # add label to name mapping for HDL evaluator

        evaluation_scores = {}
        confusion_matrix = None
        if self.class_names is None:
            evaluator = HDLEvaluator(
                es_boxes=pred_boxes,
                gt_boxes=gt_boxes,
                config=HDLEvaluationConfig(
                    overlap_condition=self.overlap_condition,
                    hard_decision_threshold=self.hard_decision_threshold,
                    f1_value_beta=self.f1_value_beta,
                ),
                with_velocity=self.with_velocity,
                origin=self.dataset_origin,
            )
            scores = evaluator.evaluate()
            evaluation_scores["all"] = scores
            confusion_matrix = evaluator.get_confusion_matrix(labels=self.class_names)
        else:
            for class_name in self.class_names + ["all"]:
                evaluator = HDLEvaluator(
                    config=self._get_hdl_eval_config(
                        class_name=class_name,
                        hdl_pedestrian_matching=self.hdl_pedestrian_matching,
                        hard_decision_threshold=self.hard_decision_threshold,
                        f1_value_beta=self.f1_value_beta,
                    ),
                    es_boxes=pred_boxes,
                    gt_boxes=gt_boxes,
                    with_velocity=self.with_velocity,
                    origin=self.dataset_origin,
                )
                scores = evaluator.evaluate()
                evaluation_scores[class_name] = scores
                if class_name == "all":
                    confusion_matrix = evaluator.get_confusion_matrix(labels=self.class_names)

        metrics_table = self._get_metrics_table(evaluation_scores)

        print_metrics_table(
            metrics_table["header"],
            metrics_table["data"],
            metrics_table["total_mAP"],
            type(self).__name__,
            logger,
        )
        if confusion_matrix is not None:
            print_log("\n" + confusion_matrix.to_markdown(), logger)

        return evaluation_scores

    def _get_metrics_table(self, evaluation_scores: dict) -> dict:
        header = list(evaluation_scores["all"].keys())
        for key in evaluation_scores:
            assert header == list(evaluation_scores[key].keys())  # all headers should be the same

        if "evaluation_conditions" in header:
            header.remove("evaluation_conditions")

        rows = []

        not_formatted = ["num-gt-boxes", "num-es-boxes", "TP", "FN", "FP"]

        for class_name in self.class_names + ["all"]:
            row = [class_name]
            for metric_name in header:
                if metric_name in not_formatted:
                    row.append(evaluation_scores[class_name][metric_name])
                else:
                    row.append("{:.3g}".format(evaluation_scores[class_name][metric_name]))
            rows.append(row)

        header = ["class"] + header
        return {"header": header, "data": rows, "total_mAP": ""}

    def _map_labels_to_names(self, labels: Sequence[int]) -> List[str]:
        ret = [self.class_names[label] for label in labels]
        return ret

    def _prepare_gt_boxes(self, gt_boxes: dict) -> dict:
        for key in gt_boxes.keys():
            new_boxes: Dict[str, Any] = {
                "bboxes_3d": [],
                "scores_3d": [],
                "detection_names": [],
            }
            for box in gt_boxes[key]:
                if self.with_velocity:
                    new_boxes["bboxes_3d"].append(torch.tensor(box["bbox_3d"] + box["velocity"]))
                else:
                    new_boxes["bboxes_3d"].append(torch.tensor(box["bbox_3d"]))
                new_boxes["scores_3d"].append(1.0)
                if self.name_mapping:
                    detection_name = self.name_mapping.map_name(
                        box["bbox_name_3d"], box["bbox_attrs_3d"]
                    )
                else:
                    detection_name = box["bbox_name_3d"]
                if self.data_mapping:
                    detection_name = self.data_mapping.get(detection_name, detection_name)
                new_boxes["detection_names"].append(detection_name)

            new_boxes["bboxes_3d"] = LiDARInstance3DBoxes(
                torch.stack(new_boxes["bboxes_3d"]),
                box_dim=9 if self.with_velocity else 7,
                origin=self.dataset_origin,
            )
            new_boxes["scores_3d"] = torch.tensor(new_boxes["scores_3d"])
            gt_boxes[key] = new_boxes
        return gt_boxes

    def _get_hdl_eval_config(
        self, class_name="car", hdl_pedestrian_matching=None, **eval_config_kwargs
    ):

        if hdl_pedestrian_matching is None:
            condition = {
                "criteria": "iou",
                "threshold": 0.5 if class_name in ["pedestrian", "bicycle"] else 0.7,
            }
        else:
            assert hdl_pedestrian_matching > 0 and hdl_pedestrian_matching < 10
            thresholds = {"pedestrian": hdl_pedestrian_matching, "bicycle": 0.5}
            condition = {
                "criteria": "iou" if class_name not in ["pedestrian"] else "distance",
                "threshold": thresholds[class_name] if class_name in thresholds else 0.7,
            }
        eval_config_dict = {
            "es_box_filter": (f'detection_names == "{class_name}"' if class_name != "all" else ""),
            "gt_box_filter": (f'detection_names == "{class_name}"' if class_name != "all" else ""),
            "overlap_condition": condition,
        }
        eval_config_dict.update(eval_config_kwargs)
        return HDLEvaluationConfig.deserialize(eval_config_dict)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample["pred_instances_3d"]
            pred_2d = data_sample["pred_instances"]
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to("cpu")
            result["pred_instances_3d"] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to("cpu")
            result["pred_instances"] = pred_2d
            sample_idx = data_sample["sample_idx"]
            result["sample_idx"] = sample_idx
            self.results.append(result)
