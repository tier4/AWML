import json
import pickle
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import numpy as np
import torch
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.dist import get_world_size
from mmengine.evaluator import BaseMetric
from mmengine.logging import MessageHub, MMLogger
from perception_eval.common import ObjectType
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel, Label, LabelType
from perception_eval.common.object import DynamicObject
from perception_eval.common.shape import Shape, ShapeType
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScore, MetricsScoreConfig
from perception_eval.evaluation.metrics.detection.ap import Ap
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
DEFAULT_T4METRIC_FILE_NAME = "t4metric_v2_results_{}.pkl"
DEFAULT_T4METRIC_METRICS_FOLDER = "metrics"


@dataclass(frozen=True)
class PerceptionFrameProcessingData:
    """Dataclass to save parameters before processing PerceptionFrameResult."""

    scene_id: str
    sample_id: str
    unix_time: float
    ground_truth_objects: FrameGroundTruth
    estimated_objects: List[ObjectType]


@METRICS.register_module()
class T4MetricV2(BaseMetric):
    """T4 format evaluation metric V2.
    Args:
        data_root (str):
            Path of dataset root.
        ann_file (str):
            Path of annotation file.
        dataset_name (str): Dataset running metrics.
        output_dir (str): Directory to save the evaluation results. Note that it's working_directory/<output_dir>.
        write_metric_summary (bool): Whether to write metric summary to json files.
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
        dataset_name: str,
        output_dir: str,
        write_metric_summary: bool,
        scene_batch_size: int = 128,
        num_workers: int = 8,
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
        self.dataset_name = dataset_name
        super(T4MetricV2, self).__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.data_root = data_root
        self.num_workers = num_workers
        self.scene_batch_size = scene_batch_size

        self.class_names = class_names
        self.name_mapping = name_mapping

        if name_mapping is not None:
            self.class_names = [self.name_mapping.get(name, name) for name in self.class_names]

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
        self.frame_results_with_info = []

        self.message_hub = MessageHub.get_current_instance()
        self.logger = MMLogger.get_current_instance()
        self.logger_file_path = Path(self.logger.log_file).parent

        # Set output directory for metrics files
        assert output_dir, f"output_dir must be provided, got: {output_dir}"
        self.output_dir = self.logger_file_path / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Metrics output directory set to: {self.output_dir}")

        self.results_pickle_path: Optional[Path] = (
            self.output_dir / results_pickle_path if results_pickle_path else None
        )
        if self.results_pickle_path and self.results_pickle_path.suffix != ".pkl":
            raise ValueError(f"results_pickle_path must end with '.pkl', got: {self.results_pickle_path}")

        self.results_pickle_exists = True if self.results_pickle_path and self.results_pickle_path.exists() else False
        self.write_metric_summary = write_metric_summary

        self.num_running_gpus = get_world_size()
        self.logger.info(f"{self.default_prefix} running with {self.num_running_gpus} GPUs")

    def evaluate(self, size: int) -> Dict[str, float]:
        """
        Evaluate the results and return a dict of metrics. Override of BaseMetric.evaluate to clean up caches
        for the multi-gpu case.
        """
        metrics = super().evaluate(size=size)
        # Clean up any caches for multi-gpu case
        self._clean_up()

        return metrics

    # override of BaseMetric.process
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model and the ground truth of dataset.
        """

        if self.results_pickle_exists:
            # Skip processing if result pickle already exists
            return

        for data_sample in data_samples:
            current_time = data_sample["timestamp"]
            scene_id = self._parse_scene_id(data_sample["lidar_path"])
            frame_ground_truth = self._parse_ground_truth_from_sample(current_time, data_sample)
            perception_frame = self._parse_predictions_from_sample(current_time, data_sample, frame_ground_truth)
            self._save_perception_frame(scene_id, data_sample["sample_idx"], perception_frame)

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
        try:
            # Load or save results based on pickle configuration
            results = self._handle_results_persistence(results)
            # Validate input
            self._validate_results(results)

            # Initialize evaluator and process scenes
            evaluator = self._create_evaluator()
            scenes = self._init_scene_from_results(results)

            # Process all frames and collect results
            self._process_all_frames(evaluator, scenes)

            # Compute final metrics
            final_metric_score = evaluator.get_scene_result()
            self.logger.info(f"Final metrics result: {final_metric_score}")
            final_metric_dict = self._process_metrics_for_aggregation(final_metric_score)

            # Write output files
            if self.write_metric_summary:
                self._write_output_files(scenes, final_metric_dict)

            return final_metric_dict

        except Exception as e:
            raise RuntimeError(f"Error in compute_metrics: {e}")
        finally:
            self._clean_up()

    def _validate_results(self, results: List[dict]) -> None:
        """Validate that the results contain valid data.

        Args:
            results (List[dict]): The results to validate.

        Raises:
            ValueError: If results are invalid.
        """
        assert results, "Results list is empty"

        assert isinstance(results, list), f"Results must be a list, got {type(results)}"

        # Check that each result is a dictionary
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                raise ValueError(f"Result at index {i} must be a dictionary, got {type(result)}")

            # Check that each result contains scene data
            if not result:
                raise ValueError(f"Result at index {i} is empty")

        self.logger.info(f"Validated {len(results)} scenes")

    def _collate_results(self, results: List[dict]) -> List[dict]:
        """Collate results from multiple GPUs.

        Args:
            results (List[dict]): List of results from different GPUs.

        Returns:
        """
        # Reinitialize
        self.scene_id_to_index_map: Dict[str, int] = {}

        # [{scene_id: {sample_id: perception_frame}}]
        tmp_results = []
        for scenes in results:
            for scene_id, samples in scenes.items():
                result_index = self.scene_id_to_index_map.get(scene_id, None)
                if result_index is not None:
                    tmp_results[result_index][scene_id].update(samples)
                else:
                    self.scene_id_to_index_map[scene_id] = len(tmp_results)
                    tmp_results.append({scene_id: samples})

        # Reorder all samples in all scenes
        for result in tmp_results:
            for scene_id, samples in result.items():
                result[scene_id] = {k: v for k, v in sorted(samples.items(), key=lambda item: item[0])}

        # Update results to the collated results
        self.results = tmp_results
        self.logger.info(f"Collated results from {len(results)} into {len(self.results)} scenes")
        return tmp_results

    def _handle_results_persistence(self, results: List[dict]) -> List[dict]:
        """Handle loading or saving results based on pickle configuration.

        Args:
            results (List[dict]): The current results.

        Returns:
            List[dict]: The results to use for evaluation.
        """
        if self.results_pickle_exists:
            self.logger.info("Loading results from pickle file")
            return self._load_results_from_pickle(self.results_pickle_path)

        # Reorganize results from multi-gpu
        if self.num_running_gpus > 1:
            results = self._collate_results(results)

        current_epoch = self.message_hub.get_info("epoch", -1) + 1
        results_pickle_path = (
            self.results_pickle_path
            if self.results_pickle_path is not None
            else self.output_dir / DEFAULT_T4METRIC_FILE_NAME.format(current_epoch)
        )
        self.logger.info(f"Saving results of epoch: {current_epoch} to pickle file: {results_pickle_path}")
        self._save_results_to_pickle(results_pickle_path)
        return results

    def _create_evaluator(self) -> PerceptionEvaluationManager:
        """Create and return a perception evaluation manager.

        Returns:
            PerceptionEvaluationManager: The configured evaluator.
        """
        metric_output_dir = self.output_dir / DEFAULT_T4METRIC_METRICS_FOLDER if self.write_metric_summary else None
        return PerceptionEvaluationManager(
            evaluation_config=self.perception_evaluator_configs,
            load_ground_truth=False,
            metric_output_dir=metric_output_dir,
        )

    def _batch_scenes(
        self, scenes: dict, scene_batch_size: int
    ) -> Generator[List[PerceptionFrameProcessingData], None, None]:
        """
        Batch scenes and group them for parallel processing based on the batch size.
        """
        batch = []
        for scene_batch_id, (scene_id, samples) in enumerate(scenes.items()):
            for sample_id, perception_frame in samples.items():
                batch.append(
                    (
                        PerceptionFrameProcessingData(
                            scene_id,
                            sample_id,
                            time.time(),
                            perception_frame.ground_truth_objects,
                            perception_frame.estimated_objects,
                        )
                    )
                )

            if (scene_batch_id + 1) % scene_batch_size == 0:
                yield batch
                batch = []

        # Any remaining batches
        if len(batch):
            yield batch

    def _parallel_preprocess_batch_frames(
        self,
        evaluator: PerceptionEvaluationManager,
        batch_index: int,
        batch_frames: List[PerceptionFrameProcessingData],
        executor: Executor,
    ) -> List[PerceptionFrameResult]:
        """
        Preprocess a batch of frames using multiprocessing.

        Args:
            evaluator (PerceptionEvaluationManager): The evaluator instance.
            batch_index (int): The index of the current batch.
            batch_frames (List[PerceptionFrameProcessingData]): List of frames in the batch.
            executor (Executor): The executor for parallel processing.

        Returns:
            List[PerceptionFrameResult]: List of preprocessed frame results.
        """
        self.logger.info(f"Pre-processing batch: {batch_index+1} with frames: {len(batch_frames)}")
        future_args = [
            (
                batch.unix_time,
                batch.ground_truth_objects,
                batch.estimated_objects,
                self.critical_object_filter_config,
                self.frame_pass_fail_config,
            )
            for batch in batch_frames
        ]

        # Unpack batched args into aligned iterables for executor.map
        (
            unix_time,
            ground_truth_objects,
            estimated_objects,
            critical_object_filter_config,
            frame_pass_fail_config,
        ) = zip(*future_args)
        # Preprocessing all frames in the batch
        perception_frame_results = list(
            executor.map(
                evaluator.preprocess_object_results,
                unix_time,
                ground_truth_objects,
                estimated_objects,
                critical_object_filter_config,
                frame_pass_fail_config,
            )
        )

        return perception_frame_results

    def _parallel_evaluate_batch_frames(
        self,
        evaluator: PerceptionEvaluationManager,
        perception_frame_results: List[PerceptionFrameResult],
        batch_index: int,
        batch_frames: List[PerceptionFrameProcessingData],
        executor: Executor,
    ) -> List[PerceptionFrameResult]:
        """
        Evaluate a batch of preprocessed PerceptionFrameResults using multiprocessing.

        Args:
            evaluator (PerceptionEvaluationManager): The evaluator instance.
            perception_frame_results (List[PerceptionFrameResult]): List of preprocessed frame results.
            batch_index (int): The index of the current batch.
            batch_frames (List[PerceptionFrameProcessingData]): List of frames in the batch.
            executor (Executor): The executor for parallel processing.
        Returns:
            List[PerceptionFrameResult]: List of evaluated frame results.
        """
        self.logger.info(f"Evaluating batch: {batch_index+1}")
        future_perception_frame_evaluation_args = [(perception_frame_results[0], None)]

        # Find the mask where an scene id is different from the previous frame, and it's the first frame of the scene
        first_sample_masks = [
            i == 0 or batch_frames[i].scene_id != batch_frames[i - 1].scene_id for i in range(len(batch_frames))
        ]

        # Group perception frame results with pair
        for index in range(1, len(perception_frame_results)):
            if first_sample_masks[index]:
                future_perception_frame_evaluation_args.append((perception_frame_results[index], None))
            else:
                future_perception_frame_evaluation_args.append(
                    (perception_frame_results[index], perception_frame_results[index - 1])
                )

        # Separate current and previous results into two sequences
        current_perception_frame_results, previous_perception_frame_results = zip(
            *future_perception_frame_evaluation_args
        )
        # Run evaluation for all frames in the batch
        perception_frame_results = list(
            executor.map(
                evaluator.evaluate_perception_frame,
                current_perception_frame_results,
                previous_perception_frame_results,
            )
        )
        return perception_frame_results

    def _postprocess_batch_frame_results(
        self,
        evaluator: PerceptionEvaluationManager,
        perception_frame_results: List[PerceptionFrameResult],
        batch_frames: List[PerceptionFrameProcessingData],
        batch_index: int,
    ) -> None:
        """Post-process the frame results.

        Args:
            frame_results (dict): The frame results to post-process.
        """
        self.logger.info(f"Post-processing batch: {batch_index+1}")
        for scene_batch, perception_frame_result in zip(batch_frames, perception_frame_results):
            # Append results
            self.frame_results_with_info.append(
                {
                    "scene_id": scene_batch.scene_id,
                    "sample_id": scene_batch.sample_id,
                    "frame_result": perception_frame_result,
                }
            )
            # We append the results outside of evaluator to keep the order of the frame results
            evaluator.frame_results.append(perception_frame_result)

    def _multi_process_all_frames(self, evaluator: PerceptionEvaluationManager, scenes: dict) -> None:
        """Process all frames in all scenes using multiprocessing to speed up frame processing.

        Args:
            evaluator (PerceptionEvaluationManager): The evaluator instance.
            scenes (dict): Dictionary of scenes and their samples.
        """
        # Multiprocessing to speed up frame processing
        self.logger.info(f"Multiprocessing with {self.num_workers} workers and batch size: {self.scene_batch_size}...")
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for batch_index, scene_batches in enumerate(
                self._batch_scenes(scenes, scene_batch_size=self.scene_batch_size)
            ):
                preprocessed_perception_frame_results = self._parallel_preprocess_batch_frames(
                    evaluator=evaluator, batch_index=batch_index, batch_frames=scene_batches, executor=executor
                )
                perception_frame_results = self._parallel_evaluate_batch_frames(
                    evaluator=evaluator,
                    perception_frame_results=preprocessed_perception_frame_results,
                    batch_index=batch_index,
                    batch_frames=scene_batches,
                    executor=executor,
                )
                self._postprocess_batch_frame_results(
                    evaluator=evaluator,
                    perception_frame_results=perception_frame_results,
                    batch_frames=scene_batches,
                    batch_index=batch_index,
                )

    def _sequential_process_all_frames(self, evaluator: PerceptionEvaluationManager, scenes: dict) -> None:
        """Process all frames in all scenes sequentially.

        Args:
              evaluator (PerceptionEvaluationManager): The evaluator instance.
              scenes (dict): Dictionary of scenes and their samples.
        """
        for scene_id, samples in scenes.items():
            for sample_id, perception_frame in samples.items():
                try:
                    frame_result: PerceptionFrameResult = evaluator.add_frame_result(
                        unix_time=time.time(),
                        ground_truth_now_frame=perception_frame.ground_truth_objects,
                        estimated_objects=perception_frame.estimated_objects,
                        critical_object_filter_config=self.critical_object_filter_config,
                        frame_pass_fail_config=self.frame_pass_fail_config,
                    )

                    self.frame_results_with_info.append(
                        {"scene_id": scene_id, "sample_id": sample_id, "frame_result": frame_result}
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to process frame {scene_id}/{sample_id}: {e}")

    def _process_all_frames(self, evaluator: PerceptionEvaluationManager, scenes: dict) -> None:
        """Process all frames in all scenes and collect frame results.

        Args:
            evaluator (PerceptionEvaluationManager): The evaluator instance.
            scenes (dict): Dictionary of scenes and their samples.
        """
        if self.num_workers > 1:
            self._multi_process_all_frames(evaluator, scenes)
        else:
            self._sequential_process_all_frames(evaluator, scenes)

    def _write_output_files(self, scenes: dict, final_metric_dict: dict) -> None:
        """Write scene metrics and aggregated metrics to files.

        Args:
            scenes (dict): Dictionary of scenes and their samples.
            final_metric_dict (dict): The final metrics dictionary.
        """
        try:
            self._write_scene_metrics(scenes)
            self._write_aggregated_metrics(final_metric_dict)
        except Exception as e:
            self.logger.error(f"Failed to write output files: {e}")

    def _clean_up(self) -> None:
        """Clean up resources after computation."""
        self.scene_id_to_index_map.clear()
        self.frame_results_with_info.clear()

    def _process_metrics_for_aggregation(self, metrics_score: MetricsScore) -> Dict[str, float]:
        """
        Process metrics from MetricsScore and return a dictionary of all metrics.

        Args:
            metrics_score (MetricsScore): The metrics score to process.

        Returns:
            Dict[str, float]: Dictionary containing all processed metrics.
        """
        metric_dict = {}

        for map_instance in metrics_score.mean_ap_values:
            matching_mode = map_instance.matching_mode.value.lower().replace(" ", "_")

            # Process individual AP values
            for label, aps in map_instance.label_to_aps.items():
                label_name = label.value

                for ap in aps:
                    threshold = ap.matching_threshold
                    ap_value = ap.ap

                    # Create the metric key
                    key = f"T4MetricV2/{label_name}_AP_{matching_mode}_{threshold}"
                    metric_dict[key] = ap_value

            # Add mAP and mAPH values
            map_key = f"T4MetricV2/mAP_{matching_mode}"
            maph_key = f"T4MetricV2/mAPH_{matching_mode}"
            metric_dict[map_key] = map_instance.map
            metric_dict[maph_key] = map_instance.maph

        return metric_dict

    def _write_aggregated_metrics(self, final_metric_dict: dict):
        """
        Writes aggregated metrics to a JSON file with the specified format.

        Args:
            final_metric_dict (dict): Dictionary containing processed metrics from the evaluator.
        """
        try:
            # Initialize the structure
            # TODO(vividf): change this when we have multiple metrics for different distance thresholds
            aggregated_metrics = {"all": {"metrics": {}, "aggregated_metric_label": {}}}

            # Organize metrics by label
            for key, value in final_metric_dict.items():
                if key.startswith("T4MetricV2/mAP_") or key.startswith("T4MetricV2/mAPH_"):
                    # These are overall metrics, put them in the metrics section
                    aggregated_metrics["all"]["metrics"][key] = value
                else:
                    # These are per-label metrics, extract label name and organize
                    # Example: T4MetricV2/car_AP_center_distance_0.5
                    parts = key.split("/")[1].split("_")
                    label_name = parts[0]  # car, truck, etc.

                    if label_name not in aggregated_metrics["all"]["aggregated_metric_label"]:
                        aggregated_metrics["all"]["aggregated_metric_label"][label_name] = {}

                    aggregated_metrics["all"]["aggregated_metric_label"][label_name][key] = value

            # Write to JSON file
            output_path = self.output_dir / "aggregated_metrics.json"
            with open(output_path, "w") as aggregated_file:
                json.dump(aggregated_metrics, aggregated_file, indent=4)

            self.logger.info(f"Aggregated metrics written to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to write aggregated metrics: {e}")
            raise

    def _write_scene_metrics(self, scenes: dict):
        """
        Writes scene metrics to a JSON file in nested format.

        Args:
            scenes (dict): Dictionary mapping scene_id to samples, where each sample contains
                          perception frame data.
        """
        try:
            # Initialize scene_metrics structure
            scene_metrics = self._initialize_scene_metrics_structure(scenes)

            # Process all frame results and populate metrics
            self._populate_scene_metrics(scene_metrics)

            # Write the nested metrics to JSON
            output_path = self.output_dir / "scene_metrics.json"
            with open(output_path, "w") as scene_file:
                json.dump(scene_metrics, scene_file, indent=4)

            self.logger.info(f"Scene metrics written to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to write scene metrics: {e}")
            raise

    def _initialize_scene_metrics_structure(self, scenes: dict) -> dict:
        """Initialize the scene metrics structure with empty dictionaries.

        Args:
            scenes (dict): Dictionary mapping scene_id to samples.

        Returns:
            dict: Initialized scene metrics structure.
        """
        return {scene_id: {sample_id: {} for sample_id in samples.keys()} for scene_id, samples in scenes.items()}

    def _populate_scene_metrics(self, scene_metrics: dict) -> None:
        """Populate scene metrics with data from frame results.

        Args:
            scene_metrics (dict): The scene metrics structure to populate.
        """
        for frame_info in self.frame_results_with_info:
            scene_id = frame_info["scene_id"]
            sample_id = frame_info["sample_id"]
            frame_result = frame_info["frame_result"]

            # Get or create the metrics structure for this frame
            frame_metrics = scene_metrics[scene_id][sample_id].setdefault("all", {})

            # Process all map instances for this frame
            self._process_frame_map_instances(frame_metrics, frame_result.metrics_score.mean_ap_values)

    def _process_frame_map_instances(self, frame_metrics: dict, map_instances) -> None:
        """Process all map instances for a single frame and populate the metrics structure.

        This method iterates through map instances (e.g., center_distance, plane_distance)
        and processes both AP (Average Precision) and APH (Average Precision with Heading)
        values for each label and threshold.

        Args:
            frame_metrics (dict): The metrics structure for this frame. This dictionary
                will be populated with the processed metrics. The structure is:
                {
                    "matching_mode1": {
                        "label_name": {
                            "ap": {"threshold": value},
                            "aph": {"threshold": value}
                        }
                    },
                    "matching_mode2": {
                        ...
                    }
                }
            map_instances: List of map instances to process. Each instance contains
                label_to_aps and label_to_aphs dictionaries.
        """
        for map_instance in map_instances:
            matching_mode = map_instance.matching_mode.value.lower().replace(" ", "_")
            matching_metrics = frame_metrics.setdefault(matching_mode, {})

            # Process AP values
            self._process_ap_values(matching_metrics, map_instance.label_to_aps)

            # Process APH values
            self._process_aph_values(matching_metrics, map_instance.label_to_aphs)

    def _process_ap_values(
        self, matching_metrics: Dict[str, Dict[str, Dict[str, float]]], label_to_aps: Dict[LabelType, List[Ap]]
    ) -> None:
        """
        Process AP values for all labels.

        Args:
            matching_metrics (Dict[str, Dict[str, Dict[str, float]]]): Nested dictionary to accumulate metrics.
                The structure is:
                    {
                        "<label_name>": {
                            "ap": {"<threshold>": <ap_value>, ...},
                            "aph": {"<threshold>": <aph_value>, ...}
                        },
                        ...
                    }
            label_to_aps (Dict[LabelType, List[Ap]]): Dictionary mapping each label
                to a list of Ap objects, each representing the AP value for a specific matching threshold.
        """
        for label, aps in label_to_aps.items():
            label_name = label.value
            label_metrics = matching_metrics.setdefault(label_name, {})
            ap_metrics = label_metrics.setdefault("ap", {})

            # Add AP values for each threshold
            for ap in aps:
                threshold_str = str(ap.matching_threshold)
                ap_metrics[threshold_str] = ap.ap

    def _process_aph_values(
        self, matching_metrics: Dict[str, Dict[str, Dict[str, float]]], label_to_aphs: Dict[LabelType, List[Ap]]
    ) -> None:
        """
        Process APH values for all labels.

        Args:
            matching_metrics (Dict[str, Dict[str, Dict[str, float]]]): Nested dictionary to accumulate metrics.
                The structure is:
                    {
                        "<label_name>": {
                            "ap": {"<threshold>": <ap_value>, ...},
                            "aph": {"<threshold>": <aph_value>, ...}
                        },
                        ...
                    }
            label_to_aphs (Dict[LabelType, List[Ap]]): Dictionary mapping each label
                to a list of Ap objects, each representing the APH value for a specific matching threshold.
        """
        for label, aphs in label_to_aphs.items():
            label_name = label.value
            label_metrics = matching_metrics.setdefault(label_name, {})
            aph_metrics = label_metrics.setdefault("aph", {})

            # Add APH values for each threshold
            for aph in aphs:
                threshold_str = str(aph.matching_threshold)
                aph_metrics[threshold_str] = aph.ap

    def _convert_index_to_label(self, bbox_label_index: int) -> Label:
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

    def _parse_scene_id(self, lidar_path: str) -> str:
        """Parse scene ID from the LiDAR file path.

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

    def _parse_ground_truth_from_sample(self, time: float, data_sample: Dict[str, Any]) -> FrameGroundTruth:
        """Parses ground truth objects from the given data sample.

        Args:
            time (float): The timestamp in seconds of the frame (sample).
            data_sample (Dict[str, Any]): A dictionary containing the ground truth data,
                                        including 3D bounding boxes, labels, and point counts.

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
                semantic_label=self._convert_index_to_label(int(label)),
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

    def _parse_predictions_from_sample(
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
                semantic_label=self._convert_index_to_label(int(label)),
            )
            for bbox, score, label in zip(bboxes, scores, labels)
            if not (np.isnan(score) or np.isnan(label) or np.any(np.isnan(bbox)))
        ]

        return PerceptionFrame(
            unix_time=time,
            estimated_objects=estimated_objects,
            ground_truth_objects=ground_truth_objects,
        )

    def _save_perception_frame(self, scene_id: str, sample_idx: int, perception_frame: PerceptionFrame) -> None:
        """
        Stores the processed perception result in self.results following the format:
        [
            {
                <scene_id>:
                    {<sample_idx>: <PerceptionFrame>},
                    {<sample_idx>: <PerceptionFrame>},
            },
            {
                <scene_id>:
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

    def _save_results_to_pickle(self, path: Path) -> None:
        """Save self.results to the given pickle file path.

        Args:
            path (Path): The full path where the pickle file will be saved.
        """
        self.logger.info(f"Saving predictions and ground truth result to pickle: {path.resolve()}")

        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.results, f)

    def _load_results_from_pickle(self, path: Path) -> List[Dict]:
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

    def _init_scene_from_results(self, results: list[Dict[str, Dict[str, Any]]]) -> dict:
        """
        Flattens scene dictionaries from the results (self.results).

        Args:
            results (list): List of dictionaries mapping scene_id to sample_id-perception_frame pairs.

        Returns:
            dict: Flattened dict of {scene_id: {sample_id: perception_frame}}.
        """
        scenes = {scene_id: samples for scene in results for scene_id, samples in scene.items()}
        return scenes
