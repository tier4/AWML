import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Sequence

# Third-party Libraries
import numpy as np
from mmdet3d.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.evaluation_task import EvaluationTask
from perception_eval.common.label import AutowareLabel, Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.shape import Shape, ShapeType

# TIER IV Perception Evaluation Library
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScoreConfig
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.manager import PerceptionEvaluationManager
from pyquaternion import Quaternion

__all__ = ["T4MetricV2"]


@METRICS.register_module()
class T4MetricV2(BaseMetric):
    """T4 format evaluation metric V2.
    Args:
        data_root (str):
            Path of dataset root.
        ann_file (str):
            Path of annotation file.
        prefix (str, optional):
            The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        collect_device (str):
            Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or 'gpu'.
            Defaults to 'cpu'.
        class_names (List[str], optional):
            The class names. Defaults to [].
        name_mapping (dict, optional):
            The data class mapping, applied to ground truth during evaluation.
            Defaults to None.
        perception_evaluator_configs (Dict[str, Any]):
            Configuration dictionary for perception evaluation.
        critical_object_filter_config (Dict[str, Any]):
            Configuration dictionary for filtering critical objects during evaluation.
        frame_pass_fail_config (Dict[str, Any]):
            Configuration dictionary that defines pass/fail criteria for perception evaluation.
        save_preds_and_gt_to_pickle (bool):
            If True, saves the predictions and groud truth to a pickle file.
            Defaults to False.
        load_preds_and_gt_from_pickle (bool):
            If True, loads the predictions and ground truth from a pickle file.
            Defaults to False.
        results_pickle_path (Optional[str]):
            Path to the pickle file used for saving or loading prediction results.
            Defaults to None.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        prefix: Optional[str] = None,
        collect_device: str = "cpu",
        class_names: List[str] = [],
        name_mapping: Optional[dict] = None,
        perception_evaluator_configs: Dict[str, Any] = {},
        critical_object_filter_config: Dict[str, Any] = {},
        frame_pass_fail_config: Dict[str, Any] = {},
        save_preds_and_gt_to_pickle: bool = False,
        load_preds_and_gt_from_pickle: bool = False,
        results_pickle_path: Optional[str] = None,
    ) -> None:

        self.default_prefix = "T4MetricV2"
        super(T4MetricV2, self).__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.data_root = data_root

        self.class_names = class_names
        self.name_mapping = name_mapping

        if name_mapping is not None:
            self.class_names = [self.name_mapping.get(name, name) for name in self.class_names]

        self.save_preds_and_gt_to_pickle = save_preds_and_gt_to_pickle
        self.load_preds_and_gt_from_pickle = load_preds_and_gt_from_pickle
        self.results_pickle_path = results_pickle_path

        self.perception_evaluator_configs: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
            **perception_evaluator_configs
        )

        self.critical_object_filter_config: CriticalObjectFilterConfig = CriticalObjectFilterConfig(
            evaluator_config=self.perception_evaluator_configs, **critical_object_filter_config
        )
        self.frame_pass_fail_config: PerceptionPassFailConfig = PerceptionPassFailConfig(
            evaluator_config=self.perception_evaluator_configs, **frame_pass_fail_config
        )

    # override of BaseMetric.process
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model and the ground truth of dataset  I am
        """

        if self.load_preds_and_gt_from_pickle:
            return

        for data_sample in data_samples:
            current_time = time.time()
            scene_id = self.parse_scene_id(data_sample["lidar_path"])
            frame_ground_truth = self.parse_ground_truth_from_sample(current_time, data_sample)
            perception_frame_result = self.parse_predictions_from_sample(current_time, data_sample, frame_ground_truth)
            self.save_perception_results(scene_id, data_sample["sample_idx"], perception_frame_result)

        if self.save_preds_and_gt_to_pickle:
            self.save_results_to_pickle(self.results_pickle_path)

    # override of BaseMetric.compute_metrics
    def compute_metrics(
        self,
        results: List[dict],
    ) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.load_preds_and_gt_from_pickle:
            logger.info("Loading predictions and ground truth result from pickle...")
            results = self.load_results_from_pickle(self.results_pickle_path)

        evaluator = PerceptionEvaluationManager(evaluation_config=self.perception_evaluator_configs)

        def process_map_instance(map_instance, metrics_store):
            matching_mode = map_instance.matching_mode.value  # e.g., "Center Distance" or "Plane Distance"

            for ap in map_instance.aps:
                label = ap.target_labels[0].value  # AutowareLabel
                threshold = ap.matching_threshold_list[0]
                ap_value = ap.ap

                # Construct the metric key
                key = f"T4MetricV2/{label}_AP_{matching_mode.lower().replace(' ', '_')}_{threshold}"
                metrics_store[key] = ap_value

        scene_metrics = {}
        for scene_dict in results:
            for scene_id, samples in scene_dict.items():
                scene_metrics.setdefault(scene_id, {})
                for sample_id, frame_result in samples.items():
                    scene_metrics[scene_id].setdefault(sample_id, {})
                    if isinstance(frame_result, PerceptionFrameResult):
                        object_with_results = frame_result.object_results
                        estimated_objects = [obj.estimated_object for obj in object_with_results]
                        frame_result: PerceptionFrameResult = evaluator.add_frame_result(
                            unix_time=time.time(),
                            ground_truth_now_frame=frame_result.frame_ground_truth,
                            estimated_objects=estimated_objects,
                            critical_object_filter_config=self.critical_object_filter_config,
                            frame_pass_fail_config=self.frame_pass_fail_config,
                        )

                        for map_instance in frame_result.metrics_score.maps:
                            process_map_instance(map_instance, scene_metrics[scene_id][sample_id])

        final_metric_score = evaluator.get_scene_result()
        logger.info(f"final metrics result {final_metric_score}")

        metric_dict = {}
        aggregated_metrics = {"aggregated_metrics": {}}

        # Iterate over the list of maps in final_metric_score
        for map_instance in final_metric_score.maps:
            process_map_instance(map_instance, metric_dict)

            for key, value in metric_dict.items():
                label = key.split("/")[1].split("_")[0]  # Extract label name
                aggregated_metrics["aggregated_metrics"].setdefault(label, {})
                aggregated_metrics["aggregated_metrics"][label][key] = value

        logger.info(f"Metric dictionary: {metric_dict}")

        with open("scene_metrics.json", "w") as scene_file:
            json.dump(scene_metrics, scene_file, indent=4)

        with open("aggregated_metrics.json", "w") as agg_file:
            json.dump(aggregated_metrics, agg_file, indent=4)

        return metric_dict

    def get_label_from_model_output(self, model_output: int) -> Label:
        """
        Retrieves a label from the model's output index

        Args:
            model_output (int): The index output by the model.

        Returns:
            Label: A Label object containing the corresponding AutowareLabel
                and the original class name.
        """
        # Check if the model output index is within the valid range
        name = self.class_names[model_output] if 0 <= model_output < len(self.class_names) else "unknown"
        autoware_label = AutowareLabel.__members__.get(name.upper(), AutowareLabel.UNKNOWN)
        return Label(label=autoware_label, name=name)

    def parse_scene_id(self, lidar_path: str) -> str:
        """parse scene ID from the LiDAR file path.

        Removes the `data_root` prefix and the trailing `/data` section.

        Args:
            lidar_path (str): The full file path of the LiDAR data.

        Returns:
            str: The extracted scene ID, or "unknown" if extraction fails.
        """
        if not lidar_path or not lidar_path.startswith(self.data_root):
            return "unknown"

        # Remove the data_root prefix
        relative_path = lidar_path[len(self.data_root) :].lstrip("/")  # Remove leading slash if exists
        path_parts = relative_path.split("/")

        # Extract scene ID before "data" section
        try:
            data_index = path_parts.index("data")
            return "/".join(path_parts[:data_index])
        except ValueError:
            return "unknown"

    def parse_ground_truth_from_sample(self, time: float, data_sample: Dict) -> FrameGroundTruth:
        """Parses ground truth objects from the given data sample.

        Args:
            time (float): The timestamp (in seconds) of the frame.
            data_sample (Dict): A dictionary containing ground truth annotations,
                                including 3D bounding boxes, labels, and LiDAR point counts.

        Returns:
            FrameGroundTruth: A structured representation of the ground truth objects,
                            including position, orientation, shape, velocity, and labels.
        """
        eval_info = data_sample.get("eval_ann_info", {})
        sample_id = data_sample.get("sample_idx", "unknown")

        gt_bboxes_3d = eval_info.get("gt_bboxes_3d", [])
        gt_labels_3d = eval_info.get("gt_labels_3d", [])
        num_lidar_pts = eval_info.get("num_lidar_pts", [])

        bboxes = (bbox.tolist() for bbox in gt_bboxes_3d.tensor.cpu().numpy())

        objects = (
            DynamicObject(
                unix_time=time,
                frame_id=self.perception_evaluator_configs.frame_id,
                position=tuple(bbox[:3]),
                orientation=Quaternion(0, 0, np.sin(bbox[6] / 2), np.cos(bbox[6] / 2)),
                shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(bbox[3:6])),
                velocity=(bbox[7], bbox[8], 0.0),
                semantic_score=1.0,
                semantic_label=self.get_label_from_model_output(int(label)),
                pointcloud_num=int(num_pts),
            )
            for bbox, label, num_pts in zip(bboxes, gt_labels_3d, num_lidar_pts)
        )

        return FrameGroundTruth(
            unix_time=time,
            frame_name=str(sample_id),
            objects=list(objects),
            transforms=None,
            raw_data=None,
        )

    def parse_predictions_from_sample(
        self, time: float, data_sample: Dict, frame_ground_truth: FrameGroundTruth
    ) -> PerceptionFrameResult:
        """
        Parses predicted objects from the data sample and creates a perception frame result.

        Args:
            time (float): The timestamp associated with the predictions.
            data_sample (Dict): A dictionary containing the predicted instances, including 3D bounding boxes, scores, and labels.
            frame_ground_truth (FrameGroundTruth): The ground truth data corresponding to the current frame.

        Returns:
            PerceptionFrameResult: A structured result containing the predicted objects, frame ground truth, and evaluation configurations.
        """
        pred_3d = data_sample.get("pred_instances_3d", {})

        bboxes = pred_3d.get("bboxes_3d", {}).tensor.cpu().numpy()
        scores = pred_3d.get("scores_3d", [])
        labels = pred_3d.get("labels_3d", [])

        # List comprehension with better clarity
        objects_with_perception = [
            DynamicObjectWithPerceptionResult(
                estimated_object=DynamicObject(
                    unix_time=time,
                    frame_id=self.perception_evaluator_configs.frame_id,
                    position=tuple(bbox[:3]),
                    orientation=Quaternion(0, 0, np.sin(bbox[6] / 2), np.cos(bbox[6] / 2)),
                    shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(bbox[3:6])),
                    velocity=(bbox[7], bbox[8], 0.0),
                    semantic_score=float(score),
                    semantic_label=self.get_label_from_model_output(int(label)),
                ),
                ground_truth_object=None,
            )
            for bbox, score, label in zip(bboxes, scores, labels)
        ]

        return PerceptionFrameResult(
            object_results=objects_with_perception,
            frame_ground_truth=frame_ground_truth,
            metrics_config=MetricsScoreConfig(
                self.perception_evaluator_configs.evaluation_task,
                target_labels={AutowareLabel[label.upper()] for label in self.class_names},
            ),
            critical_object_filter_config=self.critical_object_filter_config,
            frame_pass_fail_config=self.frame_pass_fail_config,
            unix_time=time,
            target_labels=[AutowareLabel[label.upper()] for label in self.class_names],
        )

    def save_perception_results(
        self, scene_id: str, sample_idx: int, perception_frame_result: PerceptionFrameResult
    ) -> None:
        """
        Stores the processed perceptoin result in self.results following the format.
        [
            {
                <scence_id>:
                    {<sample_idx>: <PerceptionFrameResult>},
                    {<sample_idx>: <PerceptionFrameResult>},

            },
            {
                <scence_id>:
                    {<sample_idx>: <PerceptionFrameResult>},
                    {<sample_idx>: <PerceptionFrameResult>},

            },
        ]

        Args:
            scene_id (str): The identifier for the scene to which the result belongs.
            sample_idx (int): The index of the sample within the scene.
            perception_frame_result (PerceptionFrameResult): The processed perception result for the given sample.
        """

        for scene_dict in self.results:
            if scene_id in scene_dict:
                # Append the sample to the existing scene
                scene_dict[scene_id][sample_idx] = perception_frame_result
                return

        # If scene does not exist, create a new entry
        self.results.append({scene_id: {sample_idx: perception_frame_result}})

    def save_results_to_pickle(self, path: str) -> None:
        """Save self.results to a pickle file inside the given directory.

        Args:
            path (str): The directory path where the pickle file will be stored.
        """

        if not path:
            print("[Error] Path is not specified. Please provide a valid directory path for saving results.")
            return
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "prediction_and_ground_truth.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self.results, f)

    def load_results_from_pickle(self, path: str) -> List[Dict]:
        """Loads self.results from a pickle file.

        Args:
            path (str, optional): The file path to load the pickle file from. Defaults to "prediction_and_ground_truth.pkl".

        Returns:
            List[Dict]: The deserialized results from the pickle file.
        """

        if not path:
            print("[Error] Path is not specified. Please provide a valid directory path for loading results.")
            return

        file_path = os.path.join(path, "prediction_and_ground_truth.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Pickle file not found: {file_path}")

        with open(file_path, "rb") as f:
            results = pickle.load(f)

        return results
