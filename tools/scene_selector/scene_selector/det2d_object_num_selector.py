import os
from typing import Dict, List, Set, Union

import numpy as np
import torch
from mmdet.apis import DetInferencer

from autoware_ml.registry import DATA_SELECTOR
from tools.scene_selector.scene_selector.base.image_based_scene_selector import ImageBasedSceneSelector


@DATA_SELECTOR.register_module()
class Det2dObjectNumSelector(ImageBasedSceneSelector):
    """
    A class for selecting scenes based on the number of detected objects in 2D images
    using Closed vocabulary detectors.

    This class uses a pre-trained object detection model to count specific objects
    in a batch of images and determines if the scene meets certain criteria.

    Attributes:
        valid_labels (Set[int]): Set of valid label indices.
        confidence_threshold (float): Threshold for object detection confidence.
        batch_size (int): Number of images to process in a single batch.
        classes (List[str]): List of object classes to detect.
        target_and_threshold (Dict[str, int]): Dictionary of target objects and their count thresholds.
        inferencer (DetInferencer): Object detection model for inference.
    """

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        classes: List[str],
        confidence_threshold: float = 0.3,
        target_and_threshold: Dict[str, int] = {"traffic cone": 1, "people": 20, "bicycle": 1},
        batch_size: int = 6,
    ) -> None:
        """
        Initialize the Det2dObjectNumSelector.

        Args:
            model_config_path (str): Path to the model configuration file.
            model_checkpoint_path (str): Path or URL to the model checkpoint.
            classes (List[str]): List of all detectable object classes.
            confidence_threshold (float): Confidence threshold for object detection.
            target_and_threshold (Dict[str, int]): Dictionary of target objects and their count thresholds.
            batch_size (int): Number of images to process in a single batch.
        """
        self.valid_labels = self._get_valid_labels(classes, target_and_threshold)
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.classes = classes
        self.target_and_threshold = target_and_threshold

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inferencer = DetInferencer(
            model=model_config_path,
            weights=model_checkpoint_path,
            device=device,
        )

        print("Det2dObjectNumSelector initialized with the following parameters:")
        print(f"Model Config Path: {model_config_path}")
        print(f"Model Checkpoint Path: {model_checkpoint_path}")
        print(f"Classes: {classes}")
        print(f"Confidence Threshold: {confidence_threshold}")
        print(f"Target and Threshold: {target_and_threshold}")
        print(f"Batch Size: {batch_size}")

    def _get_valid_labels(self, classes: List[str], target_and_threshold: Dict[str, int]) -> Set[int]:
        """
        Get the set of valid label indices based on the target classes.

        Args:
            classes (List[str]): List of all detectable object classes.
            target_and_threshold (Dict[str, int]): Dictionary of target objects and their count thresholds.

        Returns:
            Set[int]: Set of valid label indices.

        Raises:
            AssertionError: If a target class is not found in the list of classes.
        """
        valid_labels = set()
        for target_class in target_and_threshold:
            assert target_class in classes, f"Class {target_class} was not found in list of classes"
            valid_labels.add(classes.index(target_class))
        return valid_labels

    def is_target_scene(
        self,
        image_array: Union[List[np.ndarray], List[str]],
        return_counts: bool = False,
        results_path: str = "",
    ) -> Union[bool, tuple]:
        """
        Determine if the given images contain the target scene based on object counts.

        Args:
            image_array (Union[List[np.ndarray],List[str]]): List of images as numpy arrays.
            return_counts (bool): Whether to return the counts of the classes detected.
            results_path (str): The visualization results will be stored here, if you provide this path


        Returns:
            Union[bool, tuple]: True if the scene meets the criteria, False otherwise.
                                If return_counts is True, returns a tuple (is_target, label_counts).
        """
        results = self._get_predictions(image_array, results_path)
        label_counts = self._count_labels(results)

        is_target = any(label_counts.get(label, 0) >= thres for label, thres in self.target_and_threshold.items())

        return (is_target, {"class_counts": label_counts}) if return_counts else is_target

    def is_target_scene_multiple(
        self,
        multiple_image_arrays: List[Union[List[np.ndarray], List[str]]],
        return_counts: bool = False,
        results_path: str = "",
    ) -> List[Union[bool, tuple]]:
        """
        Determine if the given images in multiple sets contain the target scene based on object counts.

        Args:
            multiple_image_arrays (List[Union[List[np.ndarray], List[str]]]): List of lists of images as numpy arrays or file paths.
            return_counts (bool): Whether to return the counts of the classes detected.
            results_path (str): The visualization results will be stored here, if you provide this path.

        Returns:
            List[Union[bool, tuple]]: A list of True/False for each set if the scene meets the criteria,
                                    or a tuple (is_target, label_counts) for each set if return_counts is True.
        """
        flattened_images = [img for image_list in multiple_image_arrays for img in image_list]
        results = self._get_predictions(flattened_images, results_path)

        output = []
        start_idx = 0

        for image_list in multiple_image_arrays:
            end_idx = start_idx + len(image_list)
            set_results = {"predictions": results["predictions"][start_idx:end_idx]}

            label_counts = self._count_labels(set_results)
            is_target = any(label_counts.get(label, 0) >= thres for label, thres in self.target_and_threshold.items())

            if return_counts:
                output.append((is_target, label_counts))
            else:
                output.append(is_target)

            start_idx = end_idx
        if return_counts:
            return [x[0] for x in output], {"class_counts": [x[1] for x in output]}
        else:
            return output

    def _get_predictions(self, image_array: Union[List[np.ndarray], List[str]], results_path: str = "") -> Dict:
        """
        Get predictions from the object detection model.

        Args:
            image_array (Union[List[np.ndarray],List[str]]): List of images as numpy arrays.
            results_path (str): The visualization results will be stored here, if you provide this path

        Returns:
            Dict: Dictionary containing the model's predictions.
        """
        should_save = os.path.exists(results_path)
        return self.inferencer(
            inputs=image_array,
            batch_size=self.batch_size,
            no_save_vis=not should_save,
            draw_pred=should_save,
            return_datasamples=False,
            print_result=False,
            no_save_pred=True,
            out_dir=results_path,
            stuff_texts=None,
            custom_entities=False,
        )

    def _count_labels(self, results: Dict) -> Dict[str, int]:
        """
        Count the number of valid detections for each class.

        Args:
            results (Dict): Dictionary containing the model's predictions.

        Returns:
            Dict[str, int]: Dictionary of counts for each target class.
        """
        found_labels = []
        for image_result in results.get("predictions", []):
            valid_labels = [
                label
                for label, score in zip(image_result["labels"], image_result["scores"])
                if score > self.confidence_threshold and label in self.valid_labels
            ]
            found_labels.extend(valid_labels)

        unique_labels, counts = np.unique(found_labels, return_counts=True)
        return {self.classes[label]: count for label, count in zip(unique_labels, counts)}
