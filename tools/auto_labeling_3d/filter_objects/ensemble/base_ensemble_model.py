import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Type

from mmengine.registry import TASK_UTILS


@dataclass
class BaseModelInstances(ABC):
    """Base dataclass for all instances from a specific model.

    Args:
        model_id (int): Identifier for the model.
        instances (List[Dict[str, Any]]): List of instance predictions from the model.
        class_name_to_id (Dict[str, int]): Mapping from class names to class IDs.
    """

    model_id: int
    instances: List[Dict[str, Any]]
    class_name_to_id: Dict[str, int]


@TASK_UTILS.register_module()
class BaseEnsembleModel(ABC):
    """Base class for ensemble models.

    This class provides a framework for implementing various ensemble methods
    for 3D object detection results. Derived classes should implement the
    specific ensemble strategy in the ensemble_function method.

    Args:
        ensemble_setting (Dict[str, Any]): Configuration for ensembling.
        logger (logging.Logger): Logger instance.
    """

    def __init__(
        self,
        ensemble_setting: Dict[str, Any],
        logger: logging.Logger,
    ):
        self.settings = ensemble_setting
        self.logger = logger

    @property
    @abstractmethod
    def model_instances_type(self) -> Type[BaseModelInstances]:
        """Return the type of ModelInstances to use for this ensemble method."""
        pass

    @abstractmethod
    def ensemble_function(
        self,
        model_instances_list: List[BaseModelInstances],
        target_label_names: List[str],
        ensemble_settings: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Ensemble function to be implemented by derived classes.

        Args:
            model_instances_list: List of ModelInstances containing instances from each model.
            target_label_names: List of target label names.
            ensemble_settings: Dictionary containing ensemble settings.

        Returns:
            List of merged instances after ensemble.
        """
        pass

    def ensemble(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble and integrate results from all models.

        Args:
            results: List of result dictionaries, each containing metainfo and data_list.

        Returns:
            Dict[str, Any]: Ensembled results.
        """
        if len(results) == 1:
            return results[0]

        # Check if the number of weights matches the number of results
        assert len(self.settings["weights"]) == len(results), "Number of weights must match number of models"

        # Align label spaces across multiple models
        aligned_results = align_label_spaces(results)

        # Merge data_list from all models
        all_data_list: List[List[Dict[str, Any]]] = [r["data_list"] for r in aligned_results]
        class_name_to_id: Dict[str, int] = {
            class_name: class_id for class_id, class_name in enumerate(aligned_results[0]["metainfo"]["classes"])
        }
        merged_data_list: List[Dict[str, Any]] = []
        for frame_data in zip(*all_data_list):
            merged_frame = self._ensemble_frame(
                frame_data,
                ensemble_function=self.ensemble_function,
                ensemble_label_groups=self.settings["ensemble_label_groups"],
                class_name_to_id=class_name_to_id,
            )
            merged_data_list.append(merged_frame)

        return {"metainfo": aligned_results[0]["metainfo"], "data_list": merged_data_list}

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
            Dict[str, Any]: Merged frame result.
        """
        # Copy metadata from the first result
        merged_frame: Dict[str, Any] = frame_results[0].copy()
        merged_frame["instances"] = {}
        merged_instances: List[Dict[str, Any]] = []

        model_instances_list: List[BaseModelInstances] = []
        for model_idx, frame in enumerate(frame_results):
            instances: List[Dict[str, Any]] = frame.get("pred_instances_3d", [])

            model_instances_list.append(
                self.model_instances_type(
                    model_id=model_idx,
                    instances=instances,
                    weight=self.settings["weights"][model_idx],
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
                ensemble_settings=self.settings,
            )

            # All instances already have the label
            merged_instances.extend(merged_instances_by_label)

        merged_frame["pred_instances_3d"] = merged_instances
        return merged_frame


def align_label_spaces(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Align label spaces across multiple models.

    Each model has its own label space (class definitions and IDs),
    so we need to align them into a common space before ensemble.

    Args:
        results: List of results from each model.

    Returns:
        List[Dict[str, Any]]: Results with aligned label spaces.
    """
    # Merge metainfo from all models to create unified label space
    all_metainfo = [r["metainfo"] for r in results]
    merged_metainfo = _merge_class_metainfo(all_metainfo)

    # Create mapping in the unified label space
    class_name_to_id = {class_name: class_id for class_id, class_name in enumerate(merged_metainfo["classes"])}

    # Convert results to the unified label space
    aligned_results = _remap_class_ids(results, class_name_to_id)

    # Update metainfo
    for result in aligned_results:
        result["metainfo"] = merged_metainfo

    return aligned_results


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
            new_name_to_id: Dictionary mapping class names to their corresponding class IDs.

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
            new_name_to_id: Dictionary mapping class names to their corresponding class IDs.

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
