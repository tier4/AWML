import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from mmengine.registry import TASK_UTILS

from .base_ensemble_model import BaseEnsembleModel, BaseModelInstances


@dataclass
class NMSModelInstances(BaseModelInstances):
    """Dataclass for all instances from a specific model.

    Args:
        model_id (int): Identifier for the model.
        instances (List[Dict[str, Any]]): List of instance predictions from the model.
        weight (float): Weight for this model's predictions.
        class_name_to_id (Dict[str, int]): Mapping from class names to class IDs.
    """

    model_id: int
    instances: List[Dict[str, Any]]
    weight: float
    class_name_to_id: Dict[str, int]

    def filter_and_weight_instances(
        self,
        target_label_names: List[str],
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """Filter instances by label and score, and apply weight.

        Args:
            target_label_names: List of target label names.

        Returns:
            Tuple containing:
                - instances: List of instances that pass the filtering, with weighted scores.
                - boxes: Array of bounding boxes.
                - scores: Array of weighted scores.
        """
        filtered_instances = []
        boxes = []
        scores = []

        for instance in self.instances:
            # Filter by label if provided
            target_label_ids: List[int] = [
                self.class_name_to_id[target_label_name] for target_label_name in target_label_names
            ]
            if instance["bbox_label_3d"] not in target_label_ids:
                continue

            # Apply weight to score
            weighted_score = float(instance["bbox_score_3d"] * self.weight)

            # Store instance, box and score
            filtered_instances.append(instance.copy())
            boxes.append(np.array(instance["bbox_3d"]))
            scores.append(weighted_score)

        # Convert to numpy arrays if not empty
        if boxes:
            boxes = np.array(boxes)
            scores = np.array(scores)
        else:
            boxes = np.array([])
            scores = np.array([])

        return filtered_instances, boxes, scores


@TASK_UTILS.register_module()
class NMSEnsembleModel(BaseEnsembleModel):
    """A class to ensemble the results of multiple detection models using Non-Maximum Suppression (NMS).

    Args:
        ensemble_setting (Dict[str, Any]): Configuration for ensembling with:
            - weights (List[float]): Weight for each model's predictions
            - iou_threshold (float): IoU threshold for NMS
            - ensemble_label_groups (List[List[str]]): Groups of labels to ensemble
        logger (logging.Logger): Logger instance.
    """

    @property
    def model_instances_type(self) -> Type[NMSModelInstances]:
        return NMSModelInstances

    def ensemble_function(
        self,
        model_instances_list: List[NMSModelInstances],
        target_label_names: List[str],
        ensemble_settings: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """NMS-based ensemble for 3D bounding boxes.

        Args:
            model_instances_list: List of ModelInstances containing instances from each model.
            target_label_names: List of target label names.
            ensemble_settings: Dictionary containing ensemble settings:
                - iou_threshold (float): IoU threshold for NMS

        Returns:
            List of kept instances after NMS.
        """
        return _nms_ensemble(model_instances_list, target_label_names, ensemble_settings["iou_threshold"])


def _nms_ensemble(
    model_instances_list: List[NMSModelInstances],
    target_label_names: List[str],
    iou_threshold: float,
) -> List[Dict]:
    """NMS-based ensemble for 3D bounding boxes.
    Args:
        model_instances_list: List of ModelInstances containing instances from each model.
        target_label_names: List of target label names.
        iou_threshold: IoU threshold for suppression.
    Returns:
        A list of kept instances after NMS.
    """
    # Collect all filtered and weighted instances, boxes, and scores across models
    all_instances = []
    all_boxes = []
    all_scores = []

    for model_instances in model_instances_list:
        # Apply filtering and weighting with label filter
        instances, boxes, scores = model_instances.filter_and_weight_instances(target_label_names=target_label_names)

        # Add results to our collections
        if len(instances) > 0:
            all_instances.extend(instances)
            all_boxes.append(boxes)
            all_scores.append(scores)

    if not all_instances or not all_boxes:
        return []

    # Combine all boxes and scores
    boxes: np.ndarray = np.vstack(all_boxes)
    scores: np.ndarray = np.concatenate(all_scores)

    # Apply NMS
    keep_indices = _nms_indices(boxes, scores, iou_threshold)
    keep_instances = [all_instances[i] for i in keep_indices]

    return keep_instances


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Execute NMS and return indices of the boxes to keep.

    Args:
        boxes: Array of boxes [N, 7].
        scores: Array of scores [N].
        iou_threshold: IoU threshold for suppression.

    Returns:
        List of indices for boxes to keep.
    """
    order: np.ndarray = scores.argsort()[::-1]
    keep_indices: List[int] = []

    while order.size > 0:
        i = order[0]
        keep_indices.append(i)
        if order.size == 1:
            break
        remaining_boxes: np.ndarray = boxes[order[1:]]
        ious: np.ndarray = _calculate_iou(boxes[i], remaining_boxes)
        inds: np.ndarray = np.where(ious <= iou_threshold)[0]
        order: np.ndarray = order[inds + 1]

    return keep_indices


def _calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calculate IoU between a single box and an array of boxes in BEV (Bird's Eye View).

    Args:
        box: Single bounding box [x, y, z, dx, dy, dz, yaw].
        boxes: Array of bounding boxes [N, 7].

    Returns:
        Array of IoU values.
    """
    x1, y1 = box[0], box[1]
    dx1, dy1 = box[3], box[4]

    x2 = boxes[:, 0]
    y2 = boxes[:, 1]
    dx2 = boxes[:, 3]
    dy2 = boxes[:, 4]

    # Calculate overlapping area in BEV
    x_min = np.maximum(x1 - dx1 / 2, x2 - dx2 / 2)
    y_min = np.maximum(y1 - dy1 / 2, y2 - dy2 / 2)
    x_max = np.minimum(x1 + dx1 / 2, x2 + dx2 / 2)
    y_max = np.minimum(y1 + dy1 / 2, y2 + dy2 / 2)

    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    area1 = dx1 * dy1
    area2 = dx2 * dy2
    union = area1 + area2 - intersection

    return intersection / union
