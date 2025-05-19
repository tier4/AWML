import logging
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from mmengine.registry import TASK_UTILS


@TASK_UTILS.register_module()
class EnsembleModel:
    """A class to ensemble the results of multiple detection models using Non-Maximum Suppression (NMS).

    Args:
        models (List[Dict]): A list of model configurations.
        ensemble_setting (Dict[str, Any]): Configuration for ensembling (e.g., weights, iou_threshold, skip_box_threshold).
    """

    def __init__(
        self,
        ensemble_setting: Dict[str, Any],
        logger: logging.Logger,
    ):
        self.settings = ensemble_setting
        self.logger = logger

    def ensemble(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble and integrate results from all models."""
        if len(results) == 1:
            return results[0]
        # Check if the number of weights matches the number of results
        assert len(self.settings["weights"]) == len(results), "Number of weights must match number of models"

        # Obtain the union of all detected labels
        all_labels = set()
        for result in results:
            for data_info in result.get("data_list", []):
                for instance in data_info.get("pred_instances_3d", []):
                    bbox_label = instance.get("bbox_label_3d")
                    if bbox_label is not None:
                        all_labels.add(bbox_label)

        return self._ensemble(results, ensemble_function=_nms_ensemble, all_labels=all_labels)

    # TODO(Shin-kyoto): _ensembleは，どんな手法でも呼び出して使えるようにしたい．ensemble_functionに渡す引数はself.settingsとすることで抽象化したい．
    def _ensemble(self, results, ensemble_function, all_labels):
        # Initialize merged results
        merged_results = {}
        # Process each frame
        for frame_idx, frame_results in enumerate(zip(*[r["data_list"] for r in results])):
            # Copy metadata from the first result
            # TODO(Shin-kyoto): 最初のresultから，frameに関するmetadataを取得する．
            merged_frame = frame_results[0].copy()
            merged_instances = []

            # Group instances by label and ensemble
            for label in all_labels:
                boxes_list = []
                scores_list = []
                for model_idx, frame in enumerate(frame_results):
                    instances = frame.get("pred_instances_3d", [])
                    model_boxes = []
                    model_scores = []

                    for instance in instances:
                        if instance["bbox_label_3d"] == label:
                            model_boxes.append(instance["bbox_3d"])
                            model_scores.append(instance["bbox_score_3d"])

                    if model_boxes:
                        boxes_list.append(np.array(model_boxes))
                        scores_list.append(np.array(model_scores))

                if boxes_list:
                    merged_boxes, merged_scores = ensemble_function(
                        boxes_list,
                        scores_list,
                        self.settings["weights"],
                        self.settings["iou_threshold"],
                        self.settings["skip_box_threshold"],
                    )

                    # Restore the original instance information (e.g., velocity)
                    # TODO(Shin-kyoto): original boxを選ぶだけなのに，わざわざ，「一番近いものを探すためにfor文を回す」という処理を入れたくない．ensemble_functionで選んだbboxの速度を使うようにしたい．
                    for box, score in zip(merged_boxes, merged_scores):
                        original_instance = None
                        for frame in frame_results:
                            for instance in frame.get("pred_instances_3d", []):
                                if instance["bbox_label_3d"] == label and np.allclose(
                                    instance["bbox_3d"], box[:7], atol=1e-5
                                ):
                                    original_instance = instance
                                    break
                            if original_instance is not None:
                                break

                        merged_instances.append(
                            {
                                "bbox_3d": box[:7].tolist(),
                                "velocity": original_instance["velocity"] if original_instance else [0.0, 0.0],
                                "instance_id_3d": original_instance["instance_id_3d"] if original_instance else "",
                                "bbox_label_3d": label,
                                "bbox_score_3d": float(score),
                            }
                        )

            merged_frame["pred_instances_3d"] = merged_instances
            merged_results[f"frame_{frame_idx}"] = merged_frame

        return {"metainfo": results[0]["metainfo"], "data_list": list(merged_results.values())}


def _nms_ensemble(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    weights: List[float],
    iou_threshold: float,
    skip_box_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """NMS-based ensemble for 3D bounding boxes.

    Args:
        boxes_list: List of arrays containing bounding boxes from each model.
        scores_list: List of arrays containing scores from each model.
        weights: List of weights for each model.
        iou_threshold: IoU threshold for suppression.
        skip_box_threshold: Minimum score to consider a box.

    Returns:
        A tuple of arrays containing merged boxes and scores.
    """
    all_boxes = []
    all_scores = []
    # Combine the detection results from all models using weighted scores
    for model_idx, (boxes, scores) in enumerate(zip(boxes_list, scores_list)):
        for box, score in zip(boxes, scores):
            if score > skip_box_threshold:
                all_boxes.append(box)
                all_scores.append(score * weights[model_idx])

    if not all_boxes:
        return np.array([]), np.array([])

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)

    # NMS
    keep_indices = _nms_indices(all_boxes, all_scores, iou_threshold)
    keep_boxes = all_boxes[keep_indices]
    keep_scores = all_scores[keep_indices]

    return keep_boxes, keep_scores


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Execute NMS and return indices of the boxes to keep.

    Args:
        boxes: Array of boxes [N, 7].
        scores: Array of scores [N].
        iou_threshold: IoU threshold for suppression.

    Returns:
        List of indices for boxes to keep.
    """
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        remaining_boxes = boxes[order[1:]]
        ious = _calculate_iou(boxes[i], remaining_boxes)
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def _calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calculate IoU between a single box and an array of boxes in BEV (Bird's Eye View).

    Args:
        box: Single bounding box [x, y, z, dx, dy, dz, yaw].
        boxes: Array of bounding boxes [N, 7].

    Returns:
        Array of IoU values.
    """

    def calculate_bev_iou(box1, box2):
        x1, y1 = box1[0], box1[1]
        dx1, dy1 = box1[3], box1[4]

        x2, y2 = box2[0], box2[1]
        dx2, dy2 = box2[3], box2[4]

        # Calculate overlapping area in BEV
        x_min = max(x1 - dx1 / 2, x2 - dx2 / 2)
        y_min = max(y1 - dy1 / 2, y2 - dy2 / 2)
        x_max = min(x1 + dx1 / 2, x2 + dx2 / 2)
        y_max = min(y1 + dy1 / 2, y2 + dy2 / 2)

        if x_min >= x_max or y_min >= y_max:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = dx1 * dy1
        area2 = dx2 * dy2
        union = area1 + area2 - intersection

        return intersection / union

    ious = np.zeros(len(boxes))
    for i, other_box in enumerate(boxes):
        ious[i] = calculate_bev_iou(box, other_box)

    return ious
