import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel, Label
from perception_eval.common.object import DynamicObject
from perception_eval.common.shape import Shape, ShapeType
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScoreConfig
from perception_eval.evaluation.result.perception_frame import PerceptionFrame
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from perception_eval.manager import PerceptionEvaluationManager
from pyquaternion import Quaternion

__all__ = ["T4MetricV2"]
_UNKNOWN = "unknown"


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
        results_pickle_path (Optional[Union[Path, str]]):
            Path to the pickle file used for saving or loading prediction and ground truth results.

            - If not provided: runs `process()` and `compute_metrics()`.
            - If provided but the file does not exist: runs `process()` and `compute_metrics()`,
              then saves predictions and ground truth to the given path.
            - If provided and the file exists: skips `process()`, loads predictions and
              ground truth from the pickle file, and runs `compute_metrics()`.

            Defaults to None.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        prefix: Optional[str] = None,
        collect_device: str = "cpu",
        class_names: List[str] = None,
        name_mapping: Optional[dict] = None,
        perception_evaluator_configs: Optional[Dict[str, Any]] = None,
        critical_object_filter_config: Optional[Dict[str, Any]] = None,
        frame_pass_fail_config: Optional[Dict[str, Any]] = None,
        results_pickle_path: Optional[Union[Path, str]] = None,
    ) -> None:

        self.default_prefix = "T4MetricV2"
        super(T4MetricV2, self).__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.data_root = data_root

        self.class_names = class_names
        self.name_mapping = name_mapping

        if name_mapping is not None:
            self.class_names = [self.name_mapping.get(name, name) for name in self.class_names]

        self.results_pickle_path: Optional[Path] = Path(results_pickle_path) if results_pickle_path else None
        if self.results_pickle_path is not None and self.results_pickle_path.suffix != ".pkl":
            raise ValueError(f"results_pickle_path must end with '.pkl', got: {self.results_pickle_path}")

        self.results_pickle_exists = True if self.results_pickle_path and self.results_pickle_path.exists() else False

        self.target_labels = [AutowareLabel[label.upper()] for label in self.class_names]

        self.perception_evaluator_configs = PerceptionEvaluationConfig(**perception_evaluator_configs)

        self.critical_object_filter_config = CriticalObjectFilterConfig(
            evaluator_config=self.perception_evaluator_configs, **critical_object_filter_config
        )
        self.frame_pass_fail_config = PerceptionPassFailConfig(
            evaluator_config=self.perception_evaluator_configs, **frame_pass_fail_config
        )

        self.metrics_config = MetricsScoreConfig(
            self.perception_evaluator_configs.evaluation_task, target_labels=self.target_labels
        )

        self.scene_id_to_index_map: Dict[str, int] = {}  # scene_id to index map in self.results

        self.logger = MMLogger.get_current_instance()

    # override of BaseMetric.process
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model and the ground truth of dataset  I am
        """

        if self.results_pickle_exists:
            # Skip processing if result pickle already exists
            return

        for data_sample in data_samples:
            current_time = data_sample["timestamp"]
            scene_id = self.parse_scene_id(data_sample["lidar_path"])
            frame_ground_truth = self.parse_ground_truth_from_sample(current_time, data_sample)
            perception_frame = self.parse_predictions_from_sample(current_time, data_sample, frame_ground_truth)
            self.save_perception_frame(scene_id, data_sample["sample_idx"], perception_frame)

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
        Example:
            Metric dictionary:
            {
                'T4MetricV2/car_AP_center_distance_0.5': 0.7
                'T4MetricV2/truck_AP_center_distance_0.5': 0.7,
                'T4MetricV2/bus_AP_center_distance_0.5': 0.7,
                'T4MetricV2/bicycle_AP_center_distance_0.5': 0.7,
                'T4MetricV2/pedestrian_AP_center_distance_0.5': 0.7,
                ...
            }
        """

        if self.results_pickle_exists:
            results = self.load_results_from_pickle(self.results_pickle_path)
        elif self.results_pickle_path:
            self.save_results_to_pickle(self.results_pickle_path)

        evaluator = PerceptionEvaluationManager(
            evaluation_config=self.perception_evaluator_configs, load_ground_truth=False
        )

        scenes, scene_metrics = self.init_scene_metrics_from_results(results)

        for scene_id, samples in scenes.items():
            for sample_id, perception_frame in samples.items():

                frame_result: PerceptionFrameResult = evaluator.add_frame_result(
                    unix_time=time.time(),
                    ground_truth_now_frame=perception_frame.ground_truth_objects,
                    estimated_objects=perception_frame.estimated_objects,
                    critical_object_filter_config=self.critical_object_filter_config,
                    frame_pass_fail_config=self.frame_pass_fail_config,
                )

                for map_instance in frame_result.metrics_score.mean_ap_values:
                    self.process_map_instance(map_instance, scene_metrics[scene_id][sample_id])

        final_metric_score = evaluator.get_scene_result()
        self.logger.info(f"final metrics result {final_metric_score}")

        metric_dict = {}
        aggregated_metrics = {"aggregated_metrics": {}}

        # Iterate over the list of maps in final_metric_score
        for map_instance in final_metric_score.mean_ap_values:
            self.process_map_instance(map_instance, metric_dict)

            for key, value in metric_dict.items():
                # Extract label name from metric_dict's keys
                # Example: T4MetricV2/car_AP_center_distance_bev_0.5
                label = key.split("/")[1].split("_")[0]
                aggregated_metrics["aggregated_metrics"].setdefault(label, {})
                aggregated_metrics["aggregated_metrics"][label][key] = value

        with open("scene_metrics.json", "w") as scene_file:
            json.dump(scene_metrics, scene_file, indent=4)

        with open("aggregated_metrics.json", "w") as agg_file:
            json.dump(aggregated_metrics, agg_file, indent=4)

        return metric_dict

    def convert_index_to_label(self, bbox_label_index: int) -> Label:
        """
        Convert a bounding box label index into a Label object containing the corresponding AutowareLabel.

        Args:
            bbox_label_index (int): Index from the model output representing the predicted class.

        Returns:
            Label: A Label object with the corresponding AutowareLabel enum and class name string.
        """
        class_name = self.class_names[bbox_label_index] if 0 <= bbox_label_index < len(self.class_names) else _UNKNOWN
        autoware_label = AutowareLabel.__members__.get(class_name.upper(), AutowareLabel.UNKNOWN)
        return Label(label=autoware_label, name=class_name)

    def parse_scene_id(self, lidar_path: str) -> str:
        """parse scene ID from the LiDAR file path.

        Removes the `data_root` prefix and the trailing `/data` section.

        Args:
            lidar_path (str): The full file path of the LiDAR data.
            Example of the lidar_path: 'db_j6_v1/43e6e09a-93ce-488f-8f40-515187bc2753/2/data/LIDAR_CONCAT/0.pcd.bin'

        Returns:
            str: The extracted scene ID, or "unknown" if extraction fails.
            Example of the extracted scene ID: 'db_j6_v1/43e6e09a-93ce-488f-8f40-515187bc2753/2'
        """
        # TODO(vividf): This will be eventually moved to t4_devkit

        if not lidar_path or not lidar_path.startswith(self.data_root):
            return _UNKNOWN

        # Remove the data_root prefix
        relative_path = lidar_path[len(self.data_root) :].lstrip("/")  # Remove leading slash if exists
        path_parts = relative_path.split("/")

        # Extract scene ID before "data" section
        try:
            data_index = path_parts.index("data")
            return "/".join(path_parts[:data_index])
        except ValueError:
            return _UNKNOWN

    def parse_ground_truth_from_sample(self, time: float, data_sample: Dict[str, Any]) -> FrameGroundTruth:
        """Parses ground truth objects from the given data sample.

        Args:
            time (float): The timestamp in seconds of the frame (sample).
            A dictionary containing the predicted instances, including 3D bounding boxes, scores, and labels.

        Returns:
            FrameGroundTruth: A structured representation of the ground truth objects,
                            including position, orientation, shape, velocity, and labels.
        """

        # Extract evaluation annotation info for the current sample
        eval_info: dict = data_sample.get("eval_ann_info", {})
        sample_id: str = data_sample.get("sample_idx", _UNKNOWN)

        # gt_bboxes_3d: LiDARInstance3DBoxes with tensor of shape (N, 9)
        # Format per box: [x, y, z, l, w, h, yaw, vx, vy]
        gt_bboxes_3d: LiDARInstance3DBoxes = eval_info.get("gt_bboxes_3d", LiDARInstance3DBoxes([]))
        bboxes: np.ndarray = gt_bboxes_3d.tensor.cpu().numpy()

        # gt_labels_3d: (N,) array of class indices (e.g., [0, 1, 2, 3, ...])
        gt_labels_3d: np.ndarray = eval_info.get("gt_labels_3d", np.array([]))

        # num_lidar_pts: (N,) array of int, number of LiDAR points inside each GT box
        num_lidar_pts: np.ndarray = eval_info.get("num_lidar_pts", np.array([]))

        dynamic_objects = [
            DynamicObject(
                unix_time=time,
                frame_id=self.perception_evaluator_configs.frame_id,
                position=tuple(bbox[:3]),
                orientation=Quaternion(np.cos(bbox[6] / 2), 0, 0, np.sin(bbox[6] / 2)),
                shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(bbox[3:6])),
                velocity=(bbox[7], bbox[8], 0.0),
                semantic_score=1.0,
                semantic_label=self.convert_index_to_label(int(label)),
                pointcloud_num=int(num_pts),
            )
            for bbox, label, num_pts in zip(bboxes, gt_labels_3d, num_lidar_pts)
            if not (np.isnan(label) or np.isnan(num_pts) or np.any(np.isnan(bbox)))
        ]

        return FrameGroundTruth(
            unix_time=time,
            frame_name=sample_id,
            objects=dynamic_objects,
            transforms=None,
            raw_data=None,
        )

    def parse_predictions_from_sample(
        self, time: float, data_sample: Dict[str, Any], ground_truth_objects: FrameGroundTruth
    ) -> PerceptionFrame:
        """
        Parses predicted objects from the data sample and creates a perception frame result.

        Args:
            time (float): The timestamp in seconds of the frame (sample).
            data_sample (Dict[str, Any]): A dictionary containing the predicted instances, including 3D bounding boxes, scores, and labels.
            ground_truth_objects (FrameGroundTruth): The ground truth data corresponding to the current frame.

        Returns:
            PerceptionFrame: A structured result containing the predicted objects and ground truth objects.
        """
        pred_3d: Dict[str, Any] = data_sample.get("pred_instances_3d", {})

        # bboxes_3d: LiDARInstance3DBoxes with tensor of shape (N, 9)
        # Format per box: [x, y, z, l, w, h, yaw, vx, vy]
        bboxes_3d = pred_3d.get("bboxes_3d", LiDARInstance3DBoxes([]))
        bboxes: np.ndarray = bboxes_3d.tensor.cpu().numpy()

        # scores_3d: (N,) Tensor of detection confidence scores
        scores: torch.Tensor = pred_3d.get("scores_3d", torch.empty(0)).cpu()
        # labels_3d: (N,) Tensor of predicted class indices
        labels: torch.Tensor = pred_3d.get("labels_3d", torch.empty(0)).cpu()
        estimated_objects = [
            DynamicObject(
                unix_time=time,
                frame_id=self.perception_evaluator_configs.frame_id,
                position=tuple(bbox[:3]),
                orientation=Quaternion(np.cos(bbox[6] / 2), 0, 0, np.sin(bbox[6] / 2)),
                shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(bbox[3:6])),
                velocity=(bbox[7], bbox[8], 0.0),
                semantic_score=float(score),
                semantic_label=self.convert_index_to_label(int(label)),
            )
            for bbox, score, label in zip(bboxes, scores, labels)
            if not (np.isnan(score) or np.isnan(label) or np.any(np.isnan(bbox)))
        ]

        return PerceptionFrame(
            unix_time=time,
            estimated_objects=estimated_objects,
            ground_truth_objects=ground_truth_objects,
        )

    def save_perception_frame(self, scene_id: str, sample_idx: int, perception_frame: PerceptionFrame) -> None:
        """
        Stores the processed perceptoin result in self.results following the format.
        [
            {
                <scence_id>:
                    {<sample_idx>: <PerceptionFrame>},
                    {<sample_idx>: <PerceptionFrame>},

            },
            {
                <scence_id>:
                    {<sample_idx>: <PerceptionFrame>},
                    {<sample_idx>: <PerceptionFrame>},

            },
        ]

        Args:
            scene_id (str): The identifier for the scene to which the result belongs.
            sample_idx (int): The index of the sample within the scene.
            perception_frame (PerceptionFrame): The processed perception result for the given sample.
        """

        index = self.scene_id_to_index_map.get(scene_id, None)
        if index is not None:
            self.results[index][scene_id][sample_idx] = perception_frame
        else:
            # New scene: append to results and record its index
            self.results.append({scene_id: {sample_idx: perception_frame}})
            self.scene_id_to_index_map[scene_id] = len(self.results) - 1

    def save_results_to_pickle(self, path: Path) -> None:
        """Save self.results to the given pickle file path.

        Args:
            path (Path): The full path where the pickle file will be saved.
        """
        self.logger.info(f"Saving predictions and ground truth result to pickle: {path.resolve()}")

        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.results, f)

    def load_results_from_pickle(self, path: Path) -> List[Dict]:
        """Load results from a pickle file.

        Args:
            path (Path): The full path to the pickle file.

        Returns:
            List[Dict]: The deserialized results from the pickle file.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
        """
        self.logger.info(f"Loading pickle from: {path.resolve()}")
        with open(path, "rb") as f:
            results = pickle.load(f)

        return results

    def process_map_instance(self, map_instance, metrics_store):
        matching_mode = map_instance.matching_mode.value  # e.g., "Center Distance" or "Plane Distance"

        for label, aps in map_instance.label_to_aps.items():
            for ap in aps:
                threshold = ap.matching_threshold
                ap_value = ap.ap
                key = f"T4MetricV2/{label.value}_AP_{matching_mode.lower().replace(' ', '_')}_{threshold}"
                metrics_store[key] = ap_value

    def init_scene_metrics_from_results(self, results: list[Dict[str, Dict[str, Any]]]) -> tuple[dict, dict]:
        """
        Flattens scene dictionaries from the results and initializes scene_metrics structure.

        Args:
            results (list): List of dictionaries mapping scene_id to sample_id-perception_frame pairs.

        Returns:
            tuple:
                - scenes (dict): Flattened dict of {scene_id: {sample_id: perception_frame}}.
                - scene_metrics (dict): Initialized dict of {scene_id: {sample_id: dict}} for metric storage.
        """
        scenes = {scene_id: samples for scene in results for scene_id, samples in scene.items()}

        scene_metrics = {
            scene_id: {sample_id: {} for sample_id in samples.keys()} for scene_id, samples in scenes.items()
        }

        return scenes, scene_metrics
