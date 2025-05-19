import logging
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mmengine.registry import TASK_UTILS


@dataclass
class ModelInstances:
    """Dataclass for all instances from a specific model."""

    model_id: int
    instances: List[Dict[str, Any]]
    weight: float
    skip_box_threshold: float

    def filter_and_weight_instances(
        self, target_label: Optional[Any] = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """Filter instances by label and score, and apply weight.

        Args:
            target_label: If provided, only return instances with this label.

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
            if target_label is not None and instance["bbox_label_3d"] != target_label:
                continue

            # Filter by score threshold
            score = instance["bbox_score_3d"]
            if score <= self.skip_box_threshold:
                continue

            # Apply weight to score
            weighted_score = float(score * self.weight)

            # Create weighted instance
            weighted_instance = instance.copy()
            weighted_instance["bbox_score_3d"] = weighted_score

            # Store instance, box and score
            filtered_instances.append(weighted_instance)
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

        # labels for mmdet3d, e.g, [0, 1, 2, 3, 4]
        self.all_labels = [i for i, label in enumerate(self.settings["label"])]

    def ensemble(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble and integrate results from all models."""
        if len(results) == 1:
            return results[0]
        # Check if the number of weights matches the number of results
        assert len(self.settings["weights"]) == len(results), "Number of weights must match number of models"

        merged_results = []
        # Process each frame
        for frame_results in zip(*[r["data_list"] for r in results]):
            # Process single frame
            merged_frame = self._ensemble_frame(frame_results, _nms_ensemble, self.all_labels)
            merged_results.append(merged_frame)

        return {"metainfo": results[0]["metainfo"], "data_list": merged_results}

    def _ensemble_frame(self, frame_results, ensemble_function, all_labels):
        """Process a single frame's ensemble.

        Args:
            frame_results: List of results for a single frame from different models
            ensemble_function: Function to use for ensembling
            all_labels: Set of all labels across all models

        Returns:
            Merged frame data
        """
        # Copy metadata from the first result
        merged_frame = frame_results[0].copy()
        merged_instances = []

        # Group instances by label and ensemble
        for label in all_labels:
            # Collect instances from each model for this label
            model_instances_list = []
            for model_idx, frame in enumerate(frame_results):
                instances = frame.get("pred_instances_3d", [])

                model_instances_list.append(
                    ModelInstances(
                        model_id=model_idx,
                        instances=instances,
                        weight=self.settings["weights"][model_idx],
                        skip_box_threshold=self.settings["skip_box_threshold"],
                    )
                )

            if len(model_instances_list) == 0:
                raise ValueError("model_instances_list is empty")

            # Call ensemble function with instances by model
            merged_instances_by_label = ensemble_function(
                model_instances_list,
                label,
                self.settings["iou_threshold"],
            )

            # All instances already have the label
            merged_instances.extend(merged_instances_by_label)

        merged_frame["pred_instances_3d"] = merged_instances
        return merged_frame


def _nms_ensemble(
    model_instances_list: List[ModelInstances],
    label: Any,
    iou_threshold: float,
) -> List[Dict]:
    """NMS-based ensemble for 3D bounding boxes.

    Args:
        model_instances_list: List of ModelInstances containing instances from each model.
        label: The label for filtering.
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
        instances, boxes, scores = model_instances.filter_and_weight_instances(target_label=label)

        # Add results to our collections
        if len(instances) > 0:
            all_instances.extend(instances)
            all_boxes.append(boxes)
            all_scores.append(scores)

    if not all_instances or not all_boxes:
        return []

    # Combine all boxes and scores
    boxes = np.vstack(all_boxes)
    scores = np.concatenate(all_scores)

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
    order = scores.argsort()[::-1]
    keep_indices = []

    while order.size > 0:
        i = order[0]
        keep_indices.append(i)
        if order.size == 1:
            break
        remaining_boxes = boxes[order[1:]]
        ious = _calculate_iou(boxes[i], remaining_boxes)
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return keep_indices


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
