import json
import math
import pickle
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple, Union

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
    entropy_stats: Optional[dict] = None
    prediction_stats: Optional[dict] = None


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
        entropy_score_threshold (float, optional): Score threshold to include predictions in entropy
            statistics only. This does not affect evaluation metrics. Defaults to 0.3.
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
        entropy_score_threshold: Optional[float] = 0.3,
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
        self.entropy_score_threshold = entropy_score_threshold

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
        self.entropy_stats = {}
        self.prediction_stats = {}
        self._object_result_debug_written = False
        self._frame_result_debug_written = False

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
            perception_frame, entropy_stats, prediction_stats = self._parse_predictions_from_sample(
                current_time, data_sample, frame_ground_truth
            )
            self._save_perception_frame(scene_id, data_sample["sample_idx"], perception_frame)
            self._save_entropy_stats(scene_id, data_sample["sample_idx"], entropy_stats)
            self._save_prediction_stats(scene_id, data_sample["sample_idx"], prediction_stats)

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
                entropy_stats = self.entropy_stats.get(scene_id, {}).get(str(sample_id))
                prediction_stats = self.prediction_stats.get(scene_id, {}).get(str(sample_id))
                batch.append(
                    (
                        PerceptionFrameProcessingData(
                            scene_id,
                            sample_id,
                            time.time(),
                            perception_frame.ground_truth_objects,
                            perception_frame.estimated_objects,
                            entropy_stats,
                            prediction_stats,
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
                    "entropy_stats": scene_batch.entropy_stats,
                    "prediction_stats": scene_batch.prediction_stats,
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
                        {
                            "scene_id": scene_id,
                            "sample_id": sample_id,
                            "frame_result": frame_result,
                            "entropy_stats": self.entropy_stats.get(scene_id, {}).get(str(sample_id)),
                            "prediction_stats": self.prediction_stats.get(scene_id, {}).get(str(sample_id)),
                        }
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
            scene_metrics = self._write_scene_metrics(scenes)
            self._write_aggregated_metrics(final_metric_dict)
            self._write_frame_summary(scene_metrics)
            self._write_prediction_results()
        except Exception as e:
            self.logger.error(f"Failed to write output files: {e}")

    def _clean_up(self) -> None:
        """Clean up resources after computation."""
        self.scene_id_to_index_map.clear()
        self.frame_results_with_info.clear()
        self.entropy_stats.clear()
        self.prediction_stats.clear()

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

    def _write_scene_metrics(self, scenes: dict) -> dict:
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
            return scene_metrics

        except Exception as e:
            self.logger.error(f"Failed to write scene metrics: {e}")
            raise

    def _write_frame_summary(self, scene_metrics: dict) -> None:
        """Write per-frame summary with metrics and entropy in JSONL format."""
        output_path = self.output_dir / "frame_summary.jsonl"
        with open(output_path, "w") as summary_file:
            for scene_id, frames in scene_metrics.items():
                for frame_id, frame_data in frames.items():
                    record = {
                        "scene_id": scene_id,
                        "frame_id": frame_id,
                        "metrics": frame_data.get("all", {}),
                    }
                    summary_file.write(json.dumps(record) + "\n")
        self.logger.info(f"Frame summary written to: {output_path}")

    def _write_prediction_results(self) -> None:
        """Write per-prediction results with TP/FP status and entropy to JSONL."""
        output_path = self.output_dir / "prediction_results.jsonl"
        with open(output_path, "w") as pred_file:
            for frame_info in self.frame_results_with_info:
                scene_id = frame_info["scene_id"]
                frame_id = frame_info["sample_id"]
                frame_result = frame_info["frame_result"]
                prediction_stats = frame_info.get("prediction_stats", {}) or {}
                per_prediction = prediction_stats.get("per_prediction", [])

                obj_results = self._get_object_results(frame_result)
                if not obj_results and not self._frame_result_debug_written:
                    self._dump_frame_result_debug(frame_result)
                    self._frame_result_debug_written = True
                if obj_results:
                    match_map = self._match_predictions_to_object_results(obj_results, per_prediction)
                    if not match_map and len(obj_results) == len(per_prediction):
                        match_map = {idx: (idx, "index") for idx in range(len(obj_results))}
                    matched_pred_indices = set()
                    for idx, obj_result in enumerate(obj_results):
                        record = self._build_prediction_record(scene_id, frame_id, obj_result)
                        if (
                            record.get("tp_fp") is None
                            and record.get("match_type") in (None, "")
                            and not self._object_result_debug_written
                        ):
                            self._dump_object_result_debug(frame_result, obj_result)
                            self._object_result_debug_written = True
                        if idx in match_map:
                            pred_idx, alignment = match_map[idx]
                            if pred_idx < len(per_prediction):
                                record.update(per_prediction[pred_idx])
                                matched_pred_indices.add(pred_idx)
                            record["alignment"] = alignment
                            record["source"] = "combined"
                            pred_file.write(json.dumps(record) + "\n")
                        else:
                            record["alignment"] = "unmatched"
                            record["source"] = "object_result"
                            pred_file.write(json.dumps(record) + "\n")
                    for pred_idx, pred in enumerate(per_prediction):
                        if pred_idx in matched_pred_indices:
                            continue
                        pred_file.write(
                            json.dumps(
                                {
                                    "scene_id": scene_id,
                                    "frame_id": frame_id,
                                    **pred,
                                    "tp_fp": None,
                                    "match_type": None,
                                    "alignment": "unmatched",
                                    "source": "prediction",
                                }
                            )
                            + "\n"
                        )
                else:
                    # Fallback: dump entropy-only predictions without TP/FP
                    for pred in per_prediction:
                        record = {
                            "scene_id": scene_id,
                            "frame_id": frame_id,
                            **pred,
                            "tp_fp": None,
                            "match_type": None,
                            "alignment": "unmatched",
                            "source": "prediction",
                        }
                        pred_file.write(json.dumps(record) + "\n")

        self.logger.info(f"Prediction results written to: {output_path}")

    @staticmethod
    def _looks_like_object_result(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            keys = set(value.keys())
            return bool(
                keys
                & {
                    "result_type",
                    "is_true_positive",
                    "is_tp",
                    "is_false_positive",
                    "estimated_object",
                    "estimated",
                    "predicted_object",
                    "object",
                }
            )
        return any(
            hasattr(value, key)
            for key in [
                "result_type",
                "is_true_positive",
                "is_tp",
                "is_false_positive",
                "estimated_object",
                "estimated",
                "predicted_object",
                "object",
            ]
        )

    @classmethod
    def _extract_object_results(cls, value: Any, depth: int, seen: Set[int]) -> List[Any]:
        if value is None or depth > 4:
            return []
        value_id = id(value)
        if value_id in seen:
            return []
        seen.add(value_id)

        if isinstance(value, (list, tuple)):
            if value and cls._looks_like_object_result(value[0]):
                return list(value)
            for item in value:
                found = cls._extract_object_results(item, depth + 1, seen)
                if found:
                    return found
            return []

        if isinstance(value, dict):
            for item in value.values():
                found = cls._extract_object_results(item, depth + 1, seen)
                if found:
                    return found
            return []

        for attr in [
            "object_results",
            "objects_results",
            "perception_object_results",
            "result_objects",
            "object_result",
            "object_result_list",
        ]:
            if hasattr(value, attr):
                found = cls._extract_object_results(getattr(value, attr), depth + 1, seen)
                if found:
                    return found

        if hasattr(value, "__dict__"):
            for item in value.__dict__.values():
                found = cls._extract_object_results(item, depth + 1, seen)
                if found:
                    return found

        for attr in getattr(value, "__slots__", []):
            try:
                item = getattr(value, attr)
            except Exception:
                continue
            found = cls._extract_object_results(item, depth + 1, seen)
            if found:
                return found

        return []

    @classmethod
    def _get_object_results(cls, frame_result: PerceptionFrameResult) -> List[Any]:
        """Try to retrieve object results from PerceptionFrameResult across versions."""
        return cls._extract_object_results(frame_result, 0, set())

    def _dump_frame_result_debug(self, frame_result: PerceptionFrameResult) -> None:
        """Dump a summary of PerceptionFrameResult fields for debugging."""
        debug_path = self.output_dir / "frame_result_debug.json"
        debug = {
            "type": type(frame_result).__name__,
            "attrs": [],
        }

        for attr in dir(frame_result):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(frame_result, attr)
            except Exception:
                continue
            entry = {"name": attr, "type": type(value).__name__}
            if isinstance(value, (list, tuple)):
                entry["len"] = len(value)
                entry["item_type"] = type(value[0]).__name__ if value else None
                if value and self._looks_like_object_result(value[0]):
                    entry["looks_like_object_result"] = True
            elif isinstance(value, dict):
                entry["len"] = len(value)
                entry["keys"] = [str(k) for k in list(value.keys())[:5]]
            debug["attrs"].append(entry)

        try:
            with open(debug_path, "w") as debug_file:
                json.dump(debug, debug_file, indent=2)
            self.logger.info(f"Frame result debug written to: {debug_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write frame result debug info: {e}")

    def _dump_object_result_debug(self, frame_result: PerceptionFrameResult, obj_result: Any) -> None:
        """Dump a summary of object result fields for debugging TP/FP extraction."""
        debug_path = self.output_dir / "object_result_debug.json"
        debug = {
            "frame_result_type": type(frame_result).__name__,
            "object_result_type": type(obj_result).__name__,
            "object_result_attrs": [],
            "get_status": None,
            "is_result_correct": None,
            "serialization": None,
        }

        debug["get_status"] = self._safe_call_noargs(obj_result, "get_status")
        debug["is_result_correct"] = self._safe_call_noargs(obj_result, "is_result_correct")
        debug["serialization"] = self._safe_call_noargs(obj_result, "serialization")

        for attr in dir(obj_result):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(obj_result, attr)
            except Exception:
                continue
            entry = {"name": attr, "type": type(value).__name__}
            if isinstance(value, (str, int, float, bool)) or value is None:
                entry["value"] = value
            elif isinstance(value, (list, tuple)):
                entry["len"] = len(value)
                entry["item_type"] = type(value[0]).__name__ if value else None
            elif isinstance(value, dict):
                entry["len"] = len(value)
                entry["keys"] = [str(k) for k in list(value.keys())[:5]]
            else:
                entry["repr"] = repr(value)[:200]
            debug["object_result_attrs"].append(entry)

        try:
            with open(debug_path, "w") as debug_file:
                json.dump(debug, debug_file, indent=2)
            self.logger.info(f"Object result debug written to: {debug_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write object result debug info: {e}")

    @staticmethod
    def _label_to_str(label: Any) -> str:
        if label is None:
            return ""
        return getattr(label, "value", None) or getattr(label, "name", None) or str(label)

    @staticmethod
    def _safe_call_noargs(obj: Any, method_name: str) -> Optional[Any]:
        method = getattr(obj, method_name, None)
        if not callable(method):
            return None
        try:
            return method()
        except TypeError:
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_status_from_serialized(serialized: dict) -> Tuple[Optional[str], Optional[bool]]:
        match_type = None
        is_tp = None
        if not isinstance(serialized, dict):
            return match_type, is_tp

        for key in ["result_type", "status", "matching_status", "match_type", "status_label"]:
            if key in serialized and serialized[key] is not None:
                match_type = str(serialized[key])
                break

        for key in ["is_result_correct", "is_true_positive", "is_tp", "is_false_positive"]:
            if key in serialized:
                if key == "is_false_positive":
                    is_tp = not bool(serialized[key])
                else:
                    is_tp = bool(serialized[key])
                break

        if is_tp is None and "is_label_correct" in serialized:
            is_tp = bool(serialized["is_label_correct"])

        return match_type, is_tp

    @staticmethod
    def _extract_center(obj: Any) -> Optional[Tuple[float, float, float]]:
        if obj is None:
            return None
        for attr in ["position", "center", "center_position"]:
            val = getattr(obj, attr, None)
            if val is None:
                continue
            if callable(val):
                try:
                    val = val()
                except Exception:
                    continue
            try:
                coords = list(val)
            except TypeError:
                continue
            if len(coords) >= 3:
                return (float(coords[0]), float(coords[1]), float(coords[2]))

        state = getattr(obj, "state", None)
        if state is not None and state is not obj:
            center = T4MetricV2._extract_center(state)
            if center is not None:
                return center

        pose = getattr(obj, "pose", None)
        if pose is not None and pose is not obj:
            center = T4MetricV2._extract_center(pose)
            if center is not None:
                return center

        return None

    def _label_name_to_index(self, label_name: str) -> Optional[int]:
        if not label_name or not self.class_names:
            return None
        target = label_name.lower()
        for idx, name in enumerate(self.class_names):
            if name.lower() == target:
                return idx
        return None

    def _match_predictions_to_object_results(
        self,
        obj_results: List[Any],
        per_prediction: List[dict],
        center_tol: float = 1e-2,
        score_tol: float = 1e-3,
    ) -> Dict[int, Tuple[int, str]]:
        """Match object results to predictions by center/label/score similarity."""
        matches: Dict[int, Tuple[int, str]] = {}
        if not obj_results or not per_prediction:
            return matches

        pred_centers: List[Optional[Tuple[float, float, float]]] = []
        for pred in per_prediction:
            center = pred.get("center")
            if center and len(center) >= 3:
                pred_centers.append((float(center[0]), float(center[1]), float(center[2])))
            else:
                pred_centers.append(None)

        candidate_pairs: List[Tuple[float, int, int]] = []
        for obj_idx, obj_result in enumerate(obj_results):
            estimated = None
            for attr in ["estimated_object", "estimated", "predicted_object", "object"]:
                if hasattr(obj_result, attr):
                    estimated = getattr(obj_result, attr)
                    break
            if estimated is None:
                continue
            obj_center = self._extract_center(estimated)
            if obj_center is None:
                continue
            obj_score = getattr(estimated, "semantic_score", None)
            obj_label = self._label_to_str(getattr(estimated, "semantic_label", None))
            obj_label_index = self._label_name_to_index(obj_label)

            for pred_idx, pred in enumerate(per_prediction):
                pred_center = pred_centers[pred_idx]
                if pred_center is None:
                    continue
                if obj_label_index is not None:
                    pred_label_index = pred.get("label_index")
                    if pred_label_index is not None and int(pred_label_index) != obj_label_index:
                        continue
                if obj_score is not None and pred.get("score") is not None:
                    if abs(float(pred["score"]) - float(obj_score)) > score_tol:
                        continue
                dist = math.sqrt(
                    (obj_center[0] - pred_center[0]) ** 2
                    + (obj_center[1] - pred_center[1]) ** 2
                    + (obj_center[2] - pred_center[2]) ** 2
                )
                if dist <= center_tol:
                    candidate_pairs.append((dist, obj_idx, pred_idx))

        candidate_pairs.sort(key=lambda x: x[0])
        used_objs = set()
        used_preds = set()
        for _, obj_idx, pred_idx in candidate_pairs:
            if obj_idx in used_objs or pred_idx in used_preds:
                continue
            matches[obj_idx] = (pred_idx, "center")
            used_objs.add(obj_idx)
            used_preds.add(pred_idx)

        return matches

    def _build_prediction_record(self, scene_id: str, frame_id: str, obj_result: Any) -> dict:
        """Build a prediction record from an object result."""
        estimated = None
        result_type = None
        is_tp = None
        if isinstance(obj_result, dict):
            estimated = (
                obj_result.get("estimated_object")
                or obj_result.get("estimated")
                or obj_result.get("predicted_object")
                or obj_result.get("object")
            )
            result_type = obj_result.get("result_type")
            if "is_true_positive" in obj_result:
                is_tp = bool(obj_result.get("is_true_positive"))
            elif "is_tp" in obj_result:
                is_tp = bool(obj_result.get("is_tp"))
            elif "is_false_positive" in obj_result:
                is_tp = not bool(obj_result.get("is_false_positive"))
        else:
            for attr in ["estimated_object", "estimated", "predicted_object", "object"]:
                if hasattr(obj_result, attr):
                    estimated = getattr(obj_result, attr)
                    break
            result_type = getattr(obj_result, "result_type", None)
            if hasattr(obj_result, "is_true_positive"):
                is_tp = bool(getattr(obj_result, "is_true_positive"))
            elif hasattr(obj_result, "is_tp"):
                is_tp = bool(getattr(obj_result, "is_tp"))
            elif hasattr(obj_result, "is_false_positive"):
                is_tp = not bool(getattr(obj_result, "is_false_positive"))

        score = None
        label = None
        center = None
        if isinstance(estimated, dict):
            score = estimated.get("semantic_score", estimated.get("score"))
            label = estimated.get("semantic_label", estimated.get("label"))
        elif estimated is not None:
            score = getattr(estimated, "semantic_score", None)
            if score is None:
                score = getattr(estimated, "score", None)
            label = getattr(estimated, "semantic_label", None)
            if label is None:
                label = getattr(estimated, "label", None)
            center = self._extract_center(estimated)

        label = self._label_to_str(label)
        label_index = self._label_name_to_index(label)
        match_type = self._label_to_str(result_type)

        status = self._safe_call_noargs(obj_result, "get_status")
        if status is not None and not match_type:
            match_type = self._label_to_str(status)

        serialized = self._safe_call_noargs(obj_result, "serialization")
        if not match_type or is_tp is None:
            ser_match_type, ser_tp = self._extract_status_from_serialized(serialized)
            if not match_type and ser_match_type:
                match_type = ser_match_type
            if is_tp is None and ser_tp is not None:
                is_tp = ser_tp

        if is_tp is None:
            result_correct = self._safe_call_noargs(obj_result, "is_result_correct")
            if isinstance(result_correct, bool):
                is_tp = result_correct

        if is_tp is None and hasattr(obj_result, "is_label_correct"):
            is_label_correct = getattr(obj_result, "is_label_correct", None)
            gt_obj = getattr(obj_result, "ground_truth_object", None)
            if isinstance(is_label_correct, bool):
                is_tp = is_label_correct if gt_obj is not None else False

        if is_tp is None and match_type:
            mt_lower = match_type.lower()
            if "tp" in mt_lower:
                is_tp = True
            elif "fp" in mt_lower:
                is_tp = False

        record = {
            "scene_id": scene_id,
            "frame_id": frame_id,
            "score": float(score) if score is not None else None,
            "label": label,
            "label_index": label_index,
            "center": list(center) if center is not None else None,
            "tp_fp": "tp" if is_tp is True else ("fp" if is_tp is False else None),
            "match_type": match_type,
        }
        return record

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
            if frame_info.get("entropy_stats"):
                frame_metrics["entropy"] = frame_info["entropy_stats"]

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
    ) -> Tuple[PerceptionFrame, Optional[dict], Optional[dict]]:
        """
        Parses predicted objects from the data sample and creates a perception frame result.

        Args:
            time (float): The timestamp in seconds of the frame (sample).
            data_sample (Dict[str, Any]): A dictionary containing the predicted instances, including 3D bounding boxes, scores, and labels.
            ground_truth_objects (FrameGroundTruth): The ground truth data corresponding to the current frame.

        Returns:
            Tuple[PerceptionFrame, Optional[dict], Optional[dict]]: Perception frame, entropy stats,
            and per-prediction stats.
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
        class_scores: Optional[torch.Tensor] = pred_3d.get("class_scores_3d")
        entropy_stats: Optional[dict] = None
        prediction_stats: Optional[dict] = None

        estimated_objects = []
        entropies_sum: List[float] = []
        entropies_argmax: List[float] = []
        entropies_norm: List[float] = []
        entropies_sum_by_class: Dict[int, List[float]] = {}
        entropies_argmax_by_class: Dict[int, List[float]] = {}
        entropies_norm_by_class: Dict[int, List[float]] = {}
        per_prediction: List[dict] = []
        for idx, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            score_val = float(score)
            if np.isnan(score) or np.isnan(label) or np.any(np.isnan(bbox)):
                continue
            label_name = self._label_to_str(self._convert_index_to_label(int(label)))
            estimated_objects.append(
                DynamicObject(
                    unix_time=time,
                    frame_id=self.perception_evaluator_configs.frame_id,
                    position=tuple(bbox[:3]),
                    orientation=Quaternion(np.cos(bbox[6] / 2), 0, 0, np.sin(bbox[6] / 2)),
                    shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(bbox[3:6])),
                    velocity=(bbox[7], bbox[8], 0.0),
                    semantic_score=score_val,
                    semantic_label=self._convert_index_to_label(int(label)),
                )
            )
            entropy_record = {
                "index": int(idx),
                "score": score_val,
                "label_index": int(label),
                "label": label_name,
                "entropy_sum": None,
                "entropy_argmax": None,
                "entropy_normalized": None,
                "entropy_included": False,
                "entropy_reason": None,
                "center": [float(bbox[0]), float(bbox[1]), float(bbox[2])],
                "size": [float(bbox[3]), float(bbox[4]), float(bbox[5])],
                "yaw": float(bbox[6]),
            }
            if class_scores is not None and idx < class_scores.shape[0]:
                logits = class_scores[idx]
                probs = torch.sigmoid(logits)
                probs = probs.clamp_min(1e-8).clamp_max(1.0 - 1e-8)
                # Bernoulli entropy per channel, summed (log2 for bits)
                ent_sum = float(
                    (-(probs * torch.log2(probs) + (1.0 - probs) * torch.log2(1.0 - probs))).sum().cpu()
                )
                # Bernoulli entropy for argmax channel
                argmax_idx = int(torch.argmax(logits).item())
                p_arg = float(probs[argmax_idx].cpu())
                ent_arg = float(-(p_arg * np.log2(p_arg) + (1.0 - p_arg) * np.log2(1.0 - p_arg)))
                # Normalized sigmoid entropy (heuristic)
                norm = probs / probs.sum()
                ent_norm = float((-(norm * torch.log2(norm)).sum().cpu()))

                entropy_record["entropy_sum"] = ent_sum
                entropy_record["entropy_argmax"] = ent_arg
                entropy_record["entropy_normalized"] = ent_norm
                include_entropy = (
                    self.entropy_score_threshold is None or score_val >= self.entropy_score_threshold
                )
                entropy_record["entropy_included"] = include_entropy
                if not include_entropy:
                    entropy_record["entropy_reason"] = "below_score_threshold"
                if include_entropy:
                    entropies_sum_by_class.setdefault(int(label), []).append(ent_sum)
                    entropies_argmax_by_class.setdefault(int(label), []).append(ent_arg)
                    entropies_norm_by_class.setdefault(int(label), []).append(ent_norm)
                    entropies_sum.append(ent_sum)
                    entropies_argmax.append(ent_arg)
                    entropies_norm.append(ent_norm)
            else:
                entropy_record["entropy_reason"] = "missing_class_scores"
            per_prediction.append(entropy_record)

        if entropies_sum:
            per_class = {}
            for cls_idx in set(
                list(entropies_sum_by_class.keys())
                + list(entropies_argmax_by_class.keys())
                + list(entropies_norm_by_class.keys())
            ):
                if self.class_names and cls_idx < len(self.class_names):
                    cls_name = self.class_names[cls_idx]
                else:
                    cls_name = str(cls_idx)
                per_class[cls_name] = {
                    "sum": float(np.mean(entropies_sum_by_class.get(cls_idx, []))) if cls_idx in entropies_sum_by_class else None,
                    "argmax": float(np.mean(entropies_argmax_by_class.get(cls_idx, []))) if cls_idx in entropies_argmax_by_class else None,
                    "normalized": float(np.mean(entropies_norm_by_class.get(cls_idx, []))) if cls_idx in entropies_norm_by_class else None,
                }
            entropy_stats = {
                "mean_entropy_sum": float(np.mean(entropies_sum)),
                "max_entropy_sum": float(np.max(entropies_sum)),
                "mean_entropy_argmax": float(np.mean(entropies_argmax)),
                "max_entropy_argmax": float(np.max(entropies_argmax)),
                "mean_entropy_normalized": float(np.mean(entropies_norm)),
                "max_entropy_normalized": float(np.max(entropies_norm)),
                "per_class_entropy": per_class,
                "num_predictions": len(entropies_sum),
                "score_threshold": self.entropy_score_threshold,
                "entropy_log_base": 2,
            }
        if per_prediction:
            prediction_stats = {
                "per_prediction": per_prediction,
                "score_threshold": self.entropy_score_threshold,
                "entropy_log_base": 2,
            }

        return PerceptionFrame(
            unix_time=time,
            estimated_objects=estimated_objects,
            ground_truth_objects=ground_truth_objects,
        ), entropy_stats, prediction_stats

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

    def _save_entropy_stats(self, scene_id: str, sample_idx: int, entropy_stats: Optional[dict]) -> None:
        """Store entropy statistics for later export."""
        if entropy_stats is None:
            return
        scene_store = self.entropy_stats.setdefault(scene_id, {})
        scene_store[str(sample_idx)] = entropy_stats

    def _save_prediction_stats(self, scene_id: str, sample_idx: int, prediction_stats: Optional[dict]) -> None:
        """Store per-prediction stats for later export."""
        if prediction_stats is None:
            return
        scene_store = self.prediction_stats.setdefault(scene_id, {})
        scene_store[str(sample_idx)] = prediction_stats

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
