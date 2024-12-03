from tools.scene_selector.scene_selector.uncertainty_methods.mrem_uncertainty_estimator import ModelRareExampleMining
from tools.scene_selector.scene_selector.base.multi_model_scene_selector import ImagePointcloudSceneSelector
from typing import List, Dict, Union, Tuple, Optional
import numpy as np
import os
import json
from autoware_ml.registry import DATA_SELECTOR


@DATA_SELECTOR.register_module()
class MREMUncertaintyBasedSceneSelector(ImagePointcloudSceneSelector):
    """
    This class is responsible for selecting target scenes based on uncertainty estimation using the 
    ModelRareExampleMining method. It identifies scenes with rare objects based on the predictions 
    from multiple models, and applies uncertainty-based filtering.

    Parameters:
    -----------
    models : dict
        A dictionary of models with their configurations that will be used for predictions.
    iou_threshold : float, optional (default=0.4)
        The Intersection-over-Union (IoU) threshold for merging predictions from different models.
    min_score : float, optional (default=0.05)
        The minimum confidence score for a bounding box to be considered.
    p_thresh : int, optional (default=20)
        Minimum number of LiDAR points in a bounding box to be considered a valid detection.
    d_thresh : int, optional (default=75)
        Maximum distance from the sensor to consider a bounding box as valid.
    min_rare_objects : int, optional (default=2)
        The minimum number of rare objects needed for a scene to be classified as a target scene.
    rareness_threshold : float, optional (default=0.025)
        The threshold for determining whether an object is rare based on the uncertainty score. Should be in range (0,0.25)

    Attributes:
    -----------
    uncertainty_estimator : ModelRareExampleMining
        Instance of ModelRareExampleMining used to calculate uncertainty-based scores.
    min_rare_objects : int
        Minimum number of rare objects needed for a scene to be considered as a target scene.
    rareness_threshold : float
        Threshold for object rareness, above which an object is considered rare.

    Methods:
    --------
    is_target_scene(sensor_info: Union[Dict, List[Dict]], return_values: bool=False, results_path: str="") -> Union[List[bool], Tuple[List[bool], List, List, List, List, List]]:
        Determines whether the current scene contains rare objects and should be considered a target scene.
    """

    def __init__(self,
                 models: Dict,
                 iou_threshold: float = 0.5,
                 min_score: float = 0.15,
                 p_thresh: int = 20,
                 d_thresh: int = 75,
                 min_rare_objects: int = 2,
                 rareness_threshold: float = 0.025,
                 batch_size = 8) -> None:
        """
        Initializes the MREMUncertaintyBasedSceneSelector with the given model configurations and thresholds.

        Parameters:
        -----------
        models : dict
            A dictionary of models with configurations for the uncertainty estimation.
        iou_threshold : float, optional
            The IoU threshold for merging bounding boxes from different models (default: 0.4).
        min_score : float, optional
            The minimum confidence score for bounding boxes to be considered (default: 0.05).
        p_thresh : int, optional
            The minimum number of LiDAR points within a bounding box to validate the object (default: 20).
        d_thresh : int, optional
            The maximum distance from the sensor to consider the object (default: 75).
        min_rare_objects : int, optional
            The minimum number of rare objects required to consider a scene as a target (default: 2).
        rareness_threshold : float, optional
            The threshold for rareness score to define rare objects (default: 0.025).
        """
        self.uncertainty_estimator = ModelRareExampleMining(
            models, iou_threshold, min_score, p_thresh, d_thresh, batch_size)

        self.min_rare_objects = min_rare_objects
        self.rareness_threshold = rareness_threshold
        print(
            "MREMUncertaintyBasedSceneSelector initialized with the following parameters:"
        )
        print(f"Models used: {models}")
        print(f"iou_threshold: {iou_threshold}")
        print(f"min_score: {min_score}")
        print(f"p_thresh: {p_thresh}")
        print(f"d_thresh: {d_thresh}")
        print(f"min_rare_objects: {min_rare_objects}")
        print(f"rareness_threshold: {rareness_threshold}")

    def is_target_scene(self,
                        sensor_info: Union[Dict, List[Dict]],
                        return_counts: bool = False,
                        results_path: str = "") -> List[Union[bool, tuple]]:
        """
        Determines whether the current scene is a target scene based on the uncertainty scores.

        Parameters:
        -----------
        sensor_info : Union[Dict, List[Dict]]
            The sensor information containing LiDAR point cloud file paths and related data. 
            Can be a single dictionary or a list of dictionaries.
        return_values : bool, optional
            Whether to return detailed results (unique bounding boxes, data, variances, h_i, r_i) (default: False).
        results_path : str, optional
            Path to save BEV visualizations of the bounding boxes and predictions, and save results as JSON (default: "").

        Returns:
        --------
        pick_as_target : List[bool]
            A list of boolean values indicating whether the scene contains rare objects.
        unique_bboxes, data, variances, h_i, r_i : optional
            If return_values is True, the method returns additional outputs related to the bounding boxes and uncertainty estimation:
            - unique_bboxes: Bounding boxes that are unique after IoU clustering.
            - data: Uncertainty matrix for the bounding boxes from different models.
            - variances: Variance of the predictions for each bounding box.
            - h_i: Hard example filter flags for bounding boxes that passed the point count threshold.
            - r_i: Final rare example scores for each bounding box.
        """
        if isinstance(sensor_info, dict):
            sensor_info = [sensor_info]
        # Calculate the uncertainty-based scores using ModelRareExampleMining
        unique_bboxes, data, variances, h_i, r_i = self.uncertainty_estimator(
            sensor_info, results_path)

        pick_as_target: List[bool] = []
        for i, uncertainty_scores in enumerate(r_i):
            # Count the number of objects with an uncertainty score above the rareness threshold
            rare_objects_count = np.sum(
                uncertainty_scores > self.rareness_threshold)
            # If the number of rare objects exceeds the threshold, mark the scene as a target
            pick_as_target.append(rare_objects_count > self.min_rare_objects)

        # Return either the boolean decision or the detailed results
        if return_counts:
            return pick_as_target, {
                "unique_bboxes": unique_bboxes,
                "data": data,
                "variances": variances,
                "h_i": h_i,
                "r_i": r_i
            }
        else:
            return pick_as_target