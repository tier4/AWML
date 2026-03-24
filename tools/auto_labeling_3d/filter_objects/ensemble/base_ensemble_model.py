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

        self._validate_weight_count(results)
        aligned_results = align_label_spaces(results)
        class_name_to_id = self._build_class_name_to_id(aligned_results[0]["metainfo"])
        merged_data_list = self._ensemble_data_lists(aligned_results, class_name_to_id)

        return {"metainfo": aligned_results[0]["metainfo"], "data_list": merged_data_list}

    def _validate_weight_count(self, results: List[Dict[str, Any]]) -> None:
        """Validate that each model result has a matching ensemble weight."""
        assert len(self.settings["weights"]) == len(results), "Number of weights must match number of models"

    def _build_class_name_to_id(self, metainfo: Dict[str, Any]) -> Dict[str, int]:
        """Build the unified class-name to class-id mapping for ensemble."""
        return {class_name: class_id for class_id, class_name in enumerate(metainfo["classes"])}

    def _ensemble_data_lists(
        self,
        aligned_results: List[Dict[str, Any]],
        class_name_to_id: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Ensemble per-frame predictions using token alignment when available."""
        if self._can_align_by_token(aligned_results):
            return self._ensemble_token_aligned_results(aligned_results, class_name_to_id)
        return self._ensemble_positional_results(aligned_results, class_name_to_id)

    def _ensemble_token_aligned_results(
        self,
        aligned_results: List[Dict[str, Any]],
        class_name_to_id: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Ensemble frames by token so missing model outputs do not shift later frames."""
        anchor_index, anchor_data_list = self._get_anchor_result(aligned_results)
        anchor_tokens = [self._get_frame_token(frame) for frame in anchor_data_list]
        frame_lookups = [self._build_frame_lookup(result["data_list"]) for result in aligned_results]

        self._log_token_alignment_mismatches(anchor_tokens, frame_lookups, anchor_index)

        merged_data_list: List[Dict[str, Any]] = []
        for anchor_frame in anchor_data_list:
            frame_data = self._build_token_aligned_frame_data(anchor_frame, frame_lookups)
            merged_data_list.append(self._merge_frame_data(frame_data, class_name_to_id))

        return merged_data_list

    def _ensemble_positional_results(
        self,
        aligned_results: List[Dict[str, Any]],
        class_name_to_id: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Fallback to positional frame alignment when frame tokens are unavailable."""
        merged_data_list: List[Dict[str, Any]] = []
        all_data_list: List[List[Dict[str, Any]]] = [result["data_list"] for result in aligned_results]
        for frame_data in zip(*all_data_list):
            merged_data_list.append(self._merge_frame_data(frame_data, class_name_to_id))
        return merged_data_list

    def _get_anchor_result(self, aligned_results: List[Dict[str, Any]]) -> tuple[int, List[Dict[str, Any]]]:
        """Return the index and data list of the longest result stream anchor."""
        anchor_index, anchor_result = max(
            enumerate(aligned_results),
            key=lambda item: len(item[1]["data_list"]),
        )
        return anchor_index, anchor_result["data_list"]

    def _log_token_alignment_mismatches(
        self,
        anchor_tokens: List[str],
        frame_lookups: List[Dict[str, Dict[str, Any]]],
        anchor_index: int,
    ) -> None:
        """Log frames missing from or extra to the anchor sequence for non-anchor models."""
        anchor_token_set = set(anchor_tokens)
        for model_idx, frame_lookup in enumerate(frame_lookups):
            if model_idx == anchor_index:
                continue
            missing_tokens = [token for token in anchor_tokens if token not in frame_lookup]
            if missing_tokens:
                self.logger.info(
                    "Model %d is missing %d frames during ensemble; those frames will use empty predictions.",
                    model_idx,
                    len(missing_tokens),
                )

            extra_tokens = set(frame_lookup) - anchor_token_set
            if extra_tokens:
                self.logger.warning(
                    "Model %d has %d extra frames not present in the anchor result; they will be ignored.",
                    model_idx,
                    len(extra_tokens),
                )

    def _build_token_aligned_frame_data(
        self,
        anchor_frame: Dict[str, Any],
        frame_lookups: List[Dict[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Build one aligned multi-model frame tuple using the anchor frame token."""
        token = self._get_frame_token(anchor_frame)
        return [
            frame_lookup[token] if token in frame_lookup else self._generate_empty_frame(anchor_frame)
            for frame_lookup in frame_lookups
        ]

    def _merge_frame_data(
        self,
        frame_data,
        class_name_to_id: Dict[str, int],
    ) -> Dict[str, Any]:
        """Merge one frame across all models using the configured ensemble function."""
        return self._ensemble_frame(
            frame_data,
            ensemble_function=self.ensemble_function,
            ensemble_label_groups=self.settings["ensemble_label_groups"],
            class_name_to_id=class_name_to_id,
        )

    def _can_align_by_token(self, results: List[Dict[str, Any]]) -> bool:
        """Return whether all frames carry tokens and can be aligned by token."""
        return all("token" in frame for result in results for frame in result["data_list"])

    def _get_frame_token(self, frame: Dict[str, Any]) -> str:
        """Extract a frame token and raise if token-based alignment is impossible."""
        token = frame.get("token", None)
        if token is None:
            raise ValueError("Each frame must contain a token for ensemble alignment")
        return token

    def _build_frame_lookup(self, data_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build a token-to-frame lookup for one model result stream."""
        frame_lookup: Dict[str, Dict[str, Any]] = {}
        for frame in data_list:
            token = self._get_frame_token(frame)
            if token in frame_lookup:
                raise ValueError(f"Duplicate frame token found during ensemble alignment: {token}")
            frame_lookup[token] = frame
        return frame_lookup

    def _generate_empty_frame(self, reference_frame: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a metadata-preserving empty frame for a missing model prediction."""
        empty_frame = reference_frame.copy()
        empty_frame["pred_instances_3d"] = []
        return empty_frame

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
