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

    def ensemble(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble and integrate results from all models."""
        if len(results) == 1:
            return results[0]
        # Check if the number of weights matches the number of results
        assert len(self.settings["weights"]) == len(results), "Number of weights must match number of models"

        # Merge class metainfo from all models
        all_metainfo: List[Dict[str, Any]] = [r["metainfo"] for r in results]
        merged_metainfo: Dict[str, Any] = _merge_class_metainfo(all_metainfo)

        # Create mapping from class name to class id
        class_name_to_id = {class_name: class_id for class_id, class_name in enumerate(merged_metainfo["classes"])}

        # Remap class IDs in all results
        remapped_results: List[Dict[str, Any]] = _remap_class_ids(results, class_name_to_id)

        # Merge data_list from all models
        all_data_list: List[List[Dict[str, Any]]] = [r["data_list"] for r in remapped_results]
        merged_data_list: List[Dict[str, Any]] = []
        for frame_data in zip(*all_data_list):
            merged_frame = self._ensemble_frame(
                frame_data,
                ensemble_function=_nms_ensemble,
                ensemble_label_groups=self.settings["ensemble_label_groups"],
                class_name_to_id=class_name_to_id,
            )
            merged_data_list.append(merged_frame)

        return {"metainfo": merged_metainfo, "data_list": merged_data_list}

    def _ensemble_frame(
        self, frame_results, ensemble_function, ensemble_label_groups, class_name_to_id
    ) -> Dict[str, Any]:
        """Process a single frame's ensemble.

        Args:
            frame_results: List of results for a single frame from different models
            ensemble_function: Function to use for ensembling
            ensemble_label_groups: List of label name groups. Each group is processed as one ensemble unit.
                e.g. [["car", "truck", "bus"], ["pedestrian", "bicycle"]]
            class_name_to_id: Dictionary mapping class names to their corresponding class IDs.

        Returns:
            Merged frame result.
        """
        # Copy metadata from the first result
        merged_frame: Dict[str, Any] = frame_results[0].copy()
        merged_frame["instances"] = {}
        merged_instances: List[Dict[str, Any]] = []

        model_instances_list: List[ModelInstances] = []
        for model_idx, frame in enumerate(frame_results):
            instances: List[Dict[str, Any]] = frame.get("pred_instances_3d", [])

            model_instances_list.append(
                ModelInstances(
                    model_id=model_idx,
                    instances=instances,
                    weight=self.settings["weights"][model_idx],
                    skip_box_threshold=self.settings["skip_box_threshold"],
                    class_name_to_id=class_name_to_id,
                )
            )
        if len(model_instances_list) == 0:
            raise ValueError("model_instances_list is empty")

        # Group instances by label and ensemble
        for label_group in ensemble_label_groups:
            # Call ensemble function with instances by model
            merged_instances_by_label: List[Dict[str, Any]] = ensemble_function(
                model_instances_list,
                target_label_names=label_group,
                iou_threshold=self.settings["iou_threshold"],
            )

            # All instances already have the label
            merged_instances.extend(merged_instances_by_label)

        merged_frame["pred_instances_3d"] = merged_instances
        return merged_frame


def _merge_class_metainfo(metainfo_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge class metainfo from multiple models.

    Args:
        metainfo_list: List of metainfo dictionaries from multiple models.
            Each dictionary should contain 'classes' key with a list of class names
            and optionally a 'version' key.

    Returns:
        Dict[str, Any]: Merged metainfo containing combined classes and version.
            The 'classes' key contains a list of unique class names.
            The 'version' key is taken from the first metainfo if available.

    Example:
        >>> metainfo_list = [
        ...     {
        ...         'classes': ['car', 'truck', 'bus', 'bicycle', 'pedestrian'],
        ...         'version': 't4x2_pseudo'
        ...     },
        ...     {
        ...         'classes': ['cone'],
        ...         'version': 't4x2_pseudo'
        ...     }
        ... ]
        >>> _merge_class_metainfo(metainfo_list)
        {
            'classes': ['car', 'truck', 'bus', 'bicycle', 'pedestrian', 'cone'],
            'version': 't4x2_pseudo'
        }
    """
    merged_metainfo: Dict[str, Any] = {}

    # Combine all classes using set for efficient duplicate removal
    all_classes: set[str] = set()
    for metainfo in metainfo_list:
        if "classes" in metainfo:
            all_classes.update(metainfo["classes"])
    merged_metainfo["classes"] = list(all_classes)

    # Use version from the first metainfo
    if metainfo_list and "version" in metainfo_list[0]:
        merged_metainfo["version"] = metainfo_list[0]["version"]

    return merged_metainfo


def _remap_class_ids(results: List[Dict[str, Any]], new_name_to_id: Dict[str, int]) -> List[Dict[str, Any]]:
    """Remap class IDs of instances using new class name to ID mapping.

    Args:
        results: List of result dictionaries, each containing metainfo and data_list.
        new_name_to_id: Dictionary mapping class names to their corresponding class IDs.

    Returns:
        List[Dict[str, Any]]: Updated results with remapped class IDs.
    """

    def _remap_class_id_in_instance(
        instance: Dict[str, Any], old_id_to_name: Dict[int, str], new_name_to_id: Dict[str, int]
    ) -> Dict[str, Any]:
        """Remap class ID in a single instance using the new mapping.

        Args:
            instance: Instance dictionary containing bbox_label_3d.
            old_id_to_name: Dictionary mapping old class IDs to class names.

        Returns:
            Dict[str, Any]: Updated instance with remapped class ID.
        """
        converted = instance.copy()
        old_class_id = converted["bbox_label_3d"]
        class_name = old_id_to_name[old_class_id]
        converted["bbox_label_3d"] = new_name_to_id[class_name]
        return converted

    def _remap_class_ids_in_result(result: Dict[str, Any], new_name_to_id: Dict[str, int]) -> Dict[str, Any]:
        """Remap class IDs in a single result.

        Args:
            result: Result dictionary containing metainfo and data_list.

        Returns:
            Dict[str, Any]: Updated result with remapped class IDs.
        """
        # Create reverse mapping (old_id -> class_name) from result's metainfo
        old_classes: List[str] = result["metainfo"]["classes"]
        old_id_to_name: Dict[int, str] = {i: class_name for i, class_name in enumerate(old_classes)}

        updated_result = result.copy()
        updated_data_list = []

        for frame_data in result["data_list"]:
            updated_frame = frame_data.copy()
            old_instances = updated_frame.get("pred_instances_3d", [])

            # Create new instances with updated class IDs
            updated_instances = [
                _remap_class_id_in_instance(instance, old_id_to_name, new_name_to_id) for instance in old_instances
            ]

            updated_frame["pred_instances_3d"] = updated_instances
            updated_data_list.append(updated_frame)

        updated_result["metainfo"]["classes"] = list(new_name_to_id.keys())
        updated_result["data_list"] = updated_data_list
        return updated_result

    # Update class IDs in each result
    return [_remap_class_ids_in_result(result, new_name_to_id) for result in results]


def _nms_ensemble(
    model_instances_list: List[ModelInstances],
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

    def calculate_bev_iou(box1: np.ndarray, box2: np.ndarray) -> float:
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

    ious: np.ndarray = np.zeros(len(boxes))
    for i, other_box in enumerate(boxes):
        ious[i] = calculate_bev_iou(box, other_box)

    return ious
