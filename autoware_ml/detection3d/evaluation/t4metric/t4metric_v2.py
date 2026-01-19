import json
import pickle
import time
from collections import defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

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

from autoware_ml.detection3d.evaluation.t4metric.t4metric_v2_dataframe import T4MetricV2DataFrame  

__all__ = ["T4MetricV2"]
_UNKNOWN = "unknown"
DEFAULT_T4METRIC_FILE_NAME = "t4metric_v2_results_{}.pkl"
DEFAULT_T4METRIC_METRICS_FOLDER = "metrics"
DEFAULT_T4METRIC_RESULT_FOLDER = "result"


@dataclass(frozen=True)
class FrameResult:

    perception_frame_result: PerceptionFrameResult
    sample_id: str
    scene_id: int
    location: str
    vehicle_type: str


@dataclass(frozen=True)
class PerceptionFrameProcessingData:
    """Dataclass to save parameters before processing PerceptionFrameResult."""

    scene_id: str
    sample_id: str
    unix_time: float
    ground_truth_objects: FrameGroundTruth
    estimated_objects: List[ObjectType]
    perception_evaluator_manager: PerceptionEvaluationManager
    frame_pass_fail_config: PerceptionPassFailConfig
    critical_object_filter_config: Optional[CriticalObjectFilterConfig]
    evaluator_name: str
    location: str
    vehicle_type: str


@dataclass(frozen=True)
class EvaluatorData:
    """Dataclass to save data related to a PerceptionEvaluationManager."""

    perception_evaluator_manager: PerceptionEvaluationManager
    bev_distance_range: Optional[Tuple[float]]
    perception_evaluator_configs: PerceptionEvaluationConfig
    frame_pass_fail_config: PerceptionPassFailConfig
    critical_object_filter_config: Optional[CriticalObjectFilterConfig]
    metric_score_config: MetricsScoreConfig
    min_range: float
    max_range: float
    range_filter_name: str


@dataclass(frozen=True)
class PerceptionFrameMultiProcessingResult:
    """Dataclass to save data related to a PerceptionFrameResult after multiprocessing."""

    perception_frame_result: PerceptionFrameResult
    evaluator: PerceptionEvaluationManager
    scene_id: str
    sample_id: str
    evaluator_name: str
    location: str
    vehicle_type: str


def _apply_perception_evaluator_preprocessing(
    evaluator: PerceptionEvaluationManager,
    evaluator_name: str,
    scene_id: str,
    sample_id: str,
    unix_time: float,
    ground_truth_objects: List[FrameGroundTruth],
    estimated_objects: List[ObjectType],
    critical_object_filter_config: Optional[CriticalObjectFilterConfig],
    frame_pass_fail_config: PerceptionPassFailConfig,
    vehicle_type: str,
    location: str,
) -> PerceptionFrameMultiProcessingResult:
    """
    Wrapper to apply an evaluator to a list of objects for a frame in multiprocessing.

    Args:
        evaluator (PerceptionEvaluationManager): The evaluator to apply.
        evaluator_name (str): The name of the evaluator.
        scene_id (str): The scene id.
        sample_id (str): The sample id.
        unix_time (float): The unix time of the frame.
        ground_truth_objects (List[FrameGroundTruth]): The ground truth objects of the frames.
        estimated_objects (List[ObjectType]): The estimated objects of the frames.
        critical_object_filter_config (Optional[CriticalObjectFilterConfig]): The critical object filter configuration.
        frame_pass_fail_config (PerceptionPassFailConfig): The frame pass fail configuration.
    """
    # Disable visualization for multiprocessing
    evaluator.__visualizer = None

    perception_frame_result = evaluator.preprocess_object_results(
        unix_time=unix_time,
        ground_truth_now_frame=ground_truth_objects,
        estimated_objects=estimated_objects,
        critical_object_filter_config=critical_object_filter_config,
        frame_pass_fail_config=frame_pass_fail_config,
    )

    return PerceptionFrameMultiProcessingResult(
        perception_frame_result=perception_frame_result,
        evaluator=evaluator,
        scene_id=scene_id,
        sample_id=sample_id,
        evaluator_name=evaluator_name,
        vehicle_type=vehicle_type,
        location=location,
    )


def _apply_perception_evaluator_evaluation(
    evaluator: PerceptionEvaluationManager,
    evaluator_name: str,
    scene_id: str,
    sample_id: str,
    vehicle_type: str,
    location: str,
    current_perception_frame_result: PerceptionFrameResult,
    previous_perception_frame_result: Optional[PerceptionFrameResult],
) -> PerceptionFrameMultiProcessingResult:
    """
    Wrapper to apply an evaluator to a pair of PerceptionFrameResults.

    Args:
        evaluator (PerceptionEvaluationManager): The evaluator to apply.
        evaluator_name (str): The name of the evaluator.
        scene_id (str): The scene id.
        sample_id (str): The sample id.
        current_perception_frame_result (PerceptionFrameResult): The current perception frame result.
        previous_perception_frame_result (Optional[PerceptionFrameResult]): The previous perception frame result.
    """
    # Disable visualization for multiprocessing
    evaluator.__visualizer = None

    perception_frame_result = evaluator.evaluate_perception_frame(
        perception_frame_result=current_perception_frame_result,
        previous_perception_frame_result=previous_perception_frame_result,
    )
    return PerceptionFrameMultiProcessingResult(
        perception_frame_result=perception_frame_result,
        evaluator=evaluator,
        scene_id=scene_id,
        sample_id=sample_id,
        evaluator_name=evaluator_name,
        vehicle_type=vehicle_type,
        location=location,
    )


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
      bev_distance_ranges (Optional[Tuple[float]]):
        Bev distance ranges in meters for different range buckets. Defaults to None.
        Example: [(0.0, 60.0), (60.0, 90.0), (90.0, 121.0), (0.0, 121.0)], which means it will compute the metrics
        for bev distance ranges are [0.0, 60.0), [60.0, 90.0), [90.0, 121.0), [0.0, 121.0) after filtering objects by bev distance ranges, respectively.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        dataset_name: str,
        output_dir: str,
        experiment_name: str,
        experiment_group_name: str,
        write_metric_summary: bool,
        checkpoint_path: Optional[Union[Path, str]] = None,
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
        self.experiment_name = experiment_name
        self.experiment_group_name = experiment_group_name
        self.name_mapping = name_mapping
        if name_mapping is not None:
            self.class_names = [self.name_mapping.get(name, name) for name in self.class_names]

        self.target_labels = [AutowareLabel[label.upper()] for label in self.class_names]

        # scene_id to index map in self.results
        self.scene_id_to_index_map: Dict[str, int] = {}
        # {evaluator_name: []}
        self.frame_results_with_info: Dict[str, List[FrameResult]] = defaultdict(list)

        self.message_hub = MessageHub.get_current_instance()
        self.logger = MMLogger.get_current_instance()
        self.logger_file_path = Path(self.logger.log_file).parent
        self.test_timestamp = self.logger_file_path.parts[-1]
        self.checkpoint_path = checkpoint_path

        # Set output directory for metrics files
        assert output_dir, f"output_dir must be provided, got: {output_dir}"
        self.output_dir = self.logger_file_path / output_dir / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Metrics output directory set to: {self.output_dir}")

        self.results_pickle_path: Optional[Path] = results_pickle_path
        if self.results_pickle_path and self.results_pickle_path.suffix != ".pkl":
            raise ValueError(f"results_pickle_path must end with '.pkl', got: {self.results_pickle_path}")
        self.results_pickle_exists = True if self.results_pickle_path and self.results_pickle_path.exists() else False

        self.write_metric_summary = write_metric_summary
        self.num_running_gpus = get_world_size()
        self.result_output_dir = self.output_dir / DEFAULT_T4METRIC_RESULT_FOLDER
        self.evaluators = self._create_evaluators(
            perception_evaluator_configs,
            frame_pass_fail_config,
            critical_object_filter_config,
        )

        # The last evaluator is the main evaluator, which will be used to get the frame id for the ground truth
        # and predictions. Also, it's used to report the final metrics
        selected_evaluator_name = list(self.evaluators.keys())[-1]
        self.default_evaluator_prefix_name = f"{dataset_name}/{dataset_name}"
        self.main_evaluator_name = f"{self.default_evaluator_prefix_name}/{selected_evaluator_name}"
        self.main_evaluator_frame_id = self.evaluators[selected_evaluator_name].perception_evaluator_configs.frame_id
        self.logger.info(f"{self.default_prefix} running with {self.num_running_gpus} GPUs")

        # T4MetricV2 DatFrame
        self.t4_metric_v2_dataframe = T4MetricV2DataFrame(
            output_dataframe_path=self.output_dir / "t4metricv2_metrics.parquet"
        ) 

    def _create_evaluators(
        self,
        perception_evaluator_configs: Dict[str, Any],
        frame_pass_fail_configs: Dict[str, Any],
        critical_object_filter_configs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, EvaluatorData]:
        """Create and return a dictionary of evaluators.

        Returns:
            Dict[str, EvaluatorData]: A dictionary of evaluators.
        """
        # Overwrite result_output_dir with result_root_directory in perception_evaluator_configs
        perception_evaluator_configs["result_root_directory"] = str(self.result_output_dir)

        # min_distance and max_distance must be provided in perception_evaluator_configs since bev_range is mandatory
        assert (
            "min_distance" in perception_evaluator_configs["evaluation_config_dict"]
            and "max_distance" in perception_evaluator_configs["evaluation_config_dict"]
        ), "min_distance and max_distance must be provided in perception_evaluator_configs"

        assert isinstance(perception_evaluator_configs["evaluation_config_dict"]["min_distance"], list) and isinstance(
            perception_evaluator_configs["evaluation_config_dict"]["max_distance"], list
        ), f"min_distance and max_distance must be a list, got: {type(perception_evaluator_configs['evaluation_config_dict']['min_distance'])} and {type(perception_evaluator_configs['evaluation_config_dict']['max_distance'])}"

        # Form bev distance ranges from min_distance and max_distance, for example, [(min_distance[0], max_distance[0]), (min_distance[1], max_distance[1]), ...],
        # and each distance range will be used to create a separate evaluator to evaluate metrics for different bev distance ranges.
        bev_distance_ranges = []
        for min_distance, max_distance in zip(
            perception_evaluator_configs["evaluation_config_dict"]["min_distance"],
            perception_evaluator_configs["evaluation_config_dict"]["max_distance"],
        ):
            assert isinstance(min_distance, float) and isinstance(
                max_distance, float
            ), f"min_distance and max_distance must be a float, got: {type(min_distance)} and {type(max_distance)}"
            assert (
                min_distance < max_distance
            ), f"min_distance must be less than max_distance, got: {min_distance} and {max_distance}"
            bev_distance_ranges.append((min_distance, max_distance))

        range_filter_name = "bev_center"
        evaluators = {}
        for bev_distance_range in bev_distance_ranges:
            # Update min_distance_list and max_distance_list
            perception_evaluator_configs["evaluation_config_dict"]["min_distance"] = bev_distance_range[0]
            perception_evaluator_configs["evaluation_config_dict"]["max_distance"] = bev_distance_range[1]

            evaluator_config = PerceptionEvaluationConfig(**perception_evaluator_configs)
            if critical_object_filter_configs is not None:
                perception_critical_object_filter_config = CriticalObjectFilterConfig(
                    evaluator_config=evaluator_config,
                    **critical_object_filter_configs,
                )
            else:
                perception_critical_object_filter_config = None
            perception_frame_pass_fail_config = PerceptionPassFailConfig(
                evaluator_config=evaluator_config,
                **frame_pass_fail_configs,
            )
            perception_metrics_score_config = MetricsScoreConfig(
                evaluator_config.evaluation_task, target_labels=self.target_labels
            )

            evaluator_name = f"{range_filter_name}_{bev_distance_range[0]}-{bev_distance_range[1]}"
            metric_output_dir = (
                str(Path(evaluator_config.visualization_directory) / evaluator_name)
                if self.write_metric_summary
                else None
            )
            evaluator = PerceptionEvaluationManager(
                evaluation_config=evaluator_config,
                load_ground_truth=False,
                metric_output_dir=metric_output_dir,
            )
            evaluators[evaluator_name] = EvaluatorData(
                perception_evaluator_manager=evaluator,
                bev_distance_range=bev_distance_range,
                perception_evaluator_configs=evaluator_config,
                frame_pass_fail_config=perception_frame_pass_fail_config,
                critical_object_filter_config=perception_critical_object_filter_config,
                metric_score_config=perception_metrics_score_config,
                min_range=bev_distance_range[0],
                max_range=bev_distance_range[1],
                range_filter_name=range_filter_name,
            )
        return evaluators

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

    def _process_evaluator_results(
        self, scenes: dict, sample_id_to_prefix_frame_mapping: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """Process the results for each evaluator.

        Args:
            evaluator (PerceptionEvaluationManager): The evaluator instance.
            results (List[dict]): The results to process.
        """
        # Save scalar metrics and metadata only
        aggregated_metric_scalars = defaultdict(dict)

        # Save metric data, for example, detection/precisions
        aggregated_metric_data = defaultdict(dict)

        for evaluator_name, evaluator in self.evaluators.items():
            # Write scene-level metrics for each evaluator to an output file
            if self.write_metric_summary:
                try:
                    self._write_scene_metrics(scenes, evaluator_name)
                except Exception as e:
                    self.logger.error(f"Failed to write scene metrics to output files: {e}")

            # Aggregate metrics by frame prefix, for example, location and vehicle type
            frame_prefix_scores = evaluator.perception_evaluator_manager.get_scene_result_with_prefix()
            for frame_prefix_name, metric_dict in frame_prefix_scores.items():
                evaluator_frame_prefix_name = frame_prefix_name + "/" + evaluator_name

                # Process scalar metrics and metadata
                aggregated_metric_scalars[evaluator_frame_prefix_name] = self._process_metrics_for_aggregation(
                    metric_dict, evaluator_name, sample_id_to_prefix_frame_mapping
                )

                # Process metric data, for example, detection/precisions
                aggregated_metric_data[evaluator_frame_prefix_name] = self._aggregate_metrics_data(metric_dict)

            # Aggregate metrics without prefix for each evaluator
            evaluator_full_name = f"{self.default_evaluator_prefix_name}/{evaluator_name}"
            final_metric_score = evaluator.perception_evaluator_manager.get_scene_result()

            # Process scalar metrics and metadata
            aggregated_metric_scalars[evaluator_full_name] = self._process_metrics_for_aggregation(
                final_metric_score, evaluator_name, sample_id_to_prefix_frame_mapping
            )

            # Process metric data, for example, detection/precisions
            aggregated_metric_data[evaluator_full_name] = self._aggregate_metrics_data(final_metric_score)

            self.logger.info(f"====Evaluator: {evaluator_full_name}====")
            self.logger.info(f"Final metrics result: {final_metric_score}")

        # Write aggregated metrics for all evaluators to an output file
        if self.write_metric_summary:
            try:
                metric_scalars_json = self._write_aggregated_metrics(aggregated_metric_scalars, "aggregated_metrics.json")
                metric_data_json = self._write_aggregated_metrics(aggregated_metric_data, "aggregated_metrics_data.json")

                # Write to a parquet 
                df = self.t4_metric_v2_dataframe(
                    aggregated_metric_scalars=metric_scalars_json, 
                    aggregated_metric_data=metric_data_json
                )
                self.t4metric_v2_dataframe.save_dataframe(df)

            except Exception as e:
                self.logger.error(f"Failed to write aggregated metrics to output files: {e}")
         
        return aggregated_metric_scalars

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

            # Initialize scenes
            scenes = {scene_id: samples for scene in results for scene_id, samples in scene.items()}

            # Process all frames and collect results
            self._process_all_frames(scenes)

            # Construct sample_id to prefix frame mapping, assuming sample id is unique across all scenes
            sample_id_to_prefix_frame_mapping = {
                sample_id: perception_frame.ground_truth_objects.frame_prefix
                for _, samples in scenes.items()
                for sample_id, perception_frame in samples.items()
            }

            # Compute final metrics
            aggregated_metric_dict = self._process_evaluator_results(scenes, sample_id_to_prefix_frame_mapping)
            selected_aggregated_metric_dict = aggregated_metric_dict[self.main_evaluator_name]

            return selected_aggregated_metric_dict  # Return the metrics from the main evaluator

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
            self.logger.info(f"Loading results from pickle file: {self.results_pickle_path}")
            with open(self.results_pickle_path, "rb") as f:
                results = pickle.load(f)

            return results

        # Reorganize results from multi-gpu
        if self.num_running_gpus > 1:
            results = self._collate_results(results)

        # Save results to a pickle file
        current_epoch = self.message_hub.get_info("epoch", -1) + 1
        results_output_path = self.result_output_dir / DEFAULT_T4METRIC_FILE_NAME.format(current_epoch)
        self.logger.info(f"Saving results of epoch: {current_epoch} to pickle file: {results_output_path}")

        # Create parent directory if needed
        results_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_output_path, "wb") as f:
            pickle.dump(self.results, f)

        return results

    def _batch_scenes(
        self, scenes: dict, scene_batch_size: int
    ) -> Generator[List[PerceptionFrameProcessingData], None, None]:
        """
        Batch scenes and group them for parallel processing based on the batch size.
        """
        batch = []
        for scene_batch_id, (scene_id, samples) in enumerate(scenes.items()):
            # Retrieve all evaluators
            for evaluator_name, evaluator in self.evaluators.items():
                for sample_id, perception_frame in samples.items():
                    location, vehicle_type = perception_frame.ground_truth_objects.frame_prefix.split("/")
                    batch.append(
                        (
                            PerceptionFrameProcessingData(
                                scene_id=scene_id,
                                sample_id=sample_id,
                                unix_time=time.time(),
                                ground_truth_objects=perception_frame.ground_truth_objects,
                                estimated_objects=perception_frame.estimated_objects,
                                perception_evaluator_manager=evaluator.perception_evaluator_manager,
                                frame_pass_fail_config=evaluator.frame_pass_fail_config,
                                critical_object_filter_config=evaluator.critical_object_filter_config,
                                evaluator_name=evaluator_name,
                                location=location,
                                vehicle_type=vehicle_type,
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
        batch_index: int,
        batch_frames: List[PerceptionFrameProcessingData],
        executor: Executor,
    ) -> List[PerceptionFrameMultiProcessingResult]:
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
                batch.scene_id,
                batch.sample_id,
                batch.unix_time,
                batch.ground_truth_objects,
                batch.estimated_objects,
                batch.critical_object_filter_config,
                batch.frame_pass_fail_config,
                batch.perception_evaluator_manager,
                batch.evaluator_name,
                batch.vehicle_type,
                batch.location,
            )
            for batch in batch_frames
        ]

        # Unpack batched args into aligned iterables for executor.map
        (
            scene_ids,
            sample_ids,
            unix_times,
            ground_truth_objects,
            estimated_objects,
            critical_object_filter_configs,
            frame_pass_fail_configs,
            perception_evaluator_managers,
            evaluator_names,
            vehicle_types,
            locations,
        ) = zip(*future_args)

        # Preprocessing all frames in the batch
        perception_frame_preprocessing_results = list(
            executor.map(
                _apply_perception_evaluator_preprocessing,
                perception_evaluator_managers,
                evaluator_names,
                scene_ids,
                sample_ids,
                unix_times,
                ground_truth_objects,
                estimated_objects,
                critical_object_filter_configs,
                frame_pass_fail_configs,
                vehicle_types,
                locations,
            )
        )
        return perception_frame_preprocessing_results

    def _parallel_evaluate_batch_frames(
        self,
        perception_frame_preprocessing_results: List[PerceptionFrameMultiProcessingResult],
        batch_index: int,
        executor: Executor,
    ) -> List[PerceptionFrameMultiProcessingResult]:
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
        evaluation_args = []
        previous_scene_id = None
        previous_perception_frame_result = None
        previous_evaluator_name = None
        for perception_frame_preprocessing_result in perception_frame_preprocessing_results:
            # When the scene id is different from the previous frame scene id, it's the first frame of the scene or when the previous_scene_id is None
            if (
                perception_frame_preprocessing_result.scene_id != previous_scene_id
                or perception_frame_preprocessing_result.evaluator_name != previous_evaluator_name
            ):
                previous_perception_frame_result = None
                previous_scene_id = None
                previous_evaluator_name = None

            evaluation_args.append(
                (
                    perception_frame_preprocessing_result.evaluator,
                    perception_frame_preprocessing_result.evaluator_name,
                    perception_frame_preprocessing_result.scene_id,
                    perception_frame_preprocessing_result.sample_id,
                    perception_frame_preprocessing_result.vehicle_type,
                    perception_frame_preprocessing_result.location,
                    perception_frame_preprocessing_result.perception_frame_result,
                    previous_perception_frame_result,
                )
            )

            previous_perception_frame_result = perception_frame_preprocessing_result.perception_frame_result
            previous_scene_id = perception_frame_preprocessing_result.scene_id
            previous_evaluator_name = perception_frame_preprocessing_result.evaluator_name

        # Separate current and previous results into two sequences
        (
            evaluators,
            evaluator_names,
            scene_ids,
            sample_ids,
            vehicle_types,
            locations,
            current_perception_frame_results,
            previous_perception_frame_results,
        ) = zip(*evaluation_args)

        # Run evaluation for all frames in the batch
        perception_evaluation_results = list(
            executor.map(
                _apply_perception_evaluator_evaluation,
                evaluators,
                evaluator_names,
                scene_ids,
                sample_ids,
                vehicle_types,
                locations,
                current_perception_frame_results,
                previous_perception_frame_results,
            )
        )
        return perception_evaluation_results

    def _postprocess_batch_frame_results(
        self,
        perception_evaluation_results: List[PerceptionFrameMultiProcessingResult],
        batch_index: int,
    ) -> None:
        """Post-process the frame results.

        Args:
            frame_results (dict): The frame results to post-process.
        """
        self.logger.info(f"Post-processing batch: {batch_index+1}")
        for perception_evaluation_result in perception_evaluation_results:
            # Append results
            self.frame_results_with_info[perception_evaluation_result.evaluator_name].append(
                FrameResult(
                    scene_id=perception_evaluation_result.scene_id,
                    sample_id=perception_evaluation_result.sample_id,
                    location=perception_evaluation_result.location,
                    vehicle_type=perception_evaluation_result.vehicle_type,
                    perception_frame_result=perception_evaluation_result.perception_frame_result,
                )
            )

            # Since multiprocessing creates a new evaluator instance for each worker,
            # we need to append the results outside of the evaluator
            self.evaluators[
                perception_evaluation_result.evaluator_name
            ].perception_evaluator_manager.frame_results.append(perception_evaluation_result.perception_frame_result)

    def _multi_process_all_frames(self, scenes: dict) -> None:
        """Process all frames in all scenes using multiprocessing to speed up frame processing.

        Args:
            evaluator (PerceptionEvaluationManager): The evaluator instance.
            scenes (dict): Dictionary of scenes and their samples.
        """
        # Multiprocessing to speed up frame processing
        if self.scene_batch_size <= 0:
            self.scene_batch_size = len(scenes)
        self.logger.info(f"Multiprocessing with {self.num_workers} workers and batch size: {self.scene_batch_size}...")
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for batch_index, scene_batches in enumerate(
                self._batch_scenes(scenes, scene_batch_size=self.scene_batch_size)
            ):
                perception_frame_preprocessing_results = self._parallel_preprocess_batch_frames(
                    batch_index=batch_index, batch_frames=scene_batches, executor=executor
                )

                perception_evaluation_results = self._parallel_evaluate_batch_frames(
                    perception_frame_preprocessing_results=perception_frame_preprocessing_results,
                    batch_index=batch_index,
                    executor=executor,
                )

                self._postprocess_batch_frame_results(
                    perception_evaluation_results=perception_evaluation_results,
                    batch_index=batch_index,
                )

    def _sequential_process_all_frames(self, scenes: dict) -> None:
        """Process all frames in all scenes sequentially.

        Args:
              scenes (dict): Dictionary of scenes and their samples.
        """
        for evaluator_name, evaluator in self.evaluators.items():
            self.logger.info(f"Processing frames for evaluator: {evaluator_name}")
            for scene_id, samples in scenes.items():
                for sample_id, perception_frame in samples.items():
                    try:
                        location, vehicle_type = perception_frame.frame_prefix.split("/")
                        frame_result: PerceptionFrameResult = evaluator.perception_evaluator_manager.add_frame_result(
                            unix_time=time.time(),
                            ground_truth_now_frame=perception_frame.ground_truth_objects,
                            estimated_objects=perception_frame.estimated_objects,
                            critical_object_filter_config=evaluator.critical_object_filter_config,
                            frame_pass_fail_config=evaluator.frame_pass_fail_config,
                        )

                        self.frame_results_with_info[evaluator_name].append(
                            FrameResult(
                                scene_id=scene_id,
                                sample_id=sample_id,
                                location=location,
                                vehicle_type=vehicle_type,
                                perception_frame_result=frame_result,
                            )
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to process frame {scene_id}/{sample_id}: {e}")

    def _process_all_frames(self, scenes: dict) -> None:
        """Process all frames in all scenes and collect frame results.

        Args:
            evaluator (PerceptionEvaluationManager): The evaluator instance.
            scenes (dict): Dictionary of scenes and their samples.
        """
        if self.num_workers > 1:
            self._multi_process_all_frames(scenes)
        else:
            self._sequential_process_all_frames(scenes)

    def _clean_up(self) -> None:
        """Clean up resources after computation."""
        self.scene_id_to_index_map.clear()
        self.frame_results_with_info.clear()

    def _aggregate_metrics_data(
        self,
        metrics_score: MetricsScore,
    ) -> Dict[str, float]:
        """
        Process Ietarable metrics, for example, detection/precisions from MetricsScore and return a dictionary of all metrics.

        Args:
            metrics_score (MetricsScore): The metrics score to process.
            evaluator_name (str): The name of the evaluator.
            sample_id_to_prefix_frame_mapping (Dict[str, str]): A dictionary mapping sample ids to prefix frame names.

        Returns:
            Dict[str, float]: Dictionary containing all processed metrics.
        """
        iterable_metrics = {}

        total_num_preds = 0
        # Detections
        for map_instance in metrics_score.mean_ap_values:
            num_preds = 0
            matching_mode = map_instance.matching_mode.value.lower().replace(" ", "_")

            # Process individual AP values
            for label, aps in map_instance.label_to_aps.items():
                label_name = label.value

                for ap in aps:
                    threshold = ap.matching_threshold
                    ap_value = ap.ap

                    # Create precision_interpolate and recall_interpolate keys
                    iterable_metrics[
                        f"T4MetricV2_label_detection/{label_name}_precisions_{matching_mode}_{threshold}"
                    ] = ap.precision_interp.tolist()
                    iterable_metrics[
                        f"T4MetricV2_label_detection/{label_name}_recalls_{matching_mode}_{threshold}"
                    ] = ap.recall_interp.tolist()

        return iterable_metrics

    def _process_metrics_for_aggregation(
        self, metrics_score: MetricsScore, evaluator_name: str, sample_id_to_prefix_frame_mapping: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Process metrics from MetricsScore and return a dictionary of all metrics.

        Args:
            metrics_score (MetricsScore): The metrics score to process.
            evaluator_name (str): The name of the evaluator.
            sample_id_to_prefix_frame_mapping (Dict[str, str]): A dictionary mapping sample ids to prefix frame names.

        Returns:
            Dict[str, float]: Dictionary containing all processed metrics.
        """
        metric_dict = {}

        total_num_preds = 0
        for map_instance in metrics_score.mean_ap_values:
            num_preds = 0
            matching_mode = map_instance.matching_mode.value.lower().replace(" ", "_")

            # Process individual AP values
            for label, aps in map_instance.label_to_aps.items():
                label_name = label.value

                label_num_preds = aps[0].objects_results_num if len(aps) else 0
                label_num_gts = map_instance.num_ground_truth_dict.get(label, 0) if len(aps) else 0
                num_preds += label_num_preds
                for ap in aps:
                    threshold = ap.matching_threshold
                    ap_value = ap.ap

                    # Create the metric key
                    key = f"T4MetricV2_label/{label_name}_AP_{matching_mode}_{threshold}"
                    metric_dict[key] = ap_value

                    # Create max f1_score key
                    metric_dict[f"T4MetricV2_label/{label_name}_max-f1score_{matching_mode}_{threshold}"] = (
                        ap.max_f1_score
                    )

                    # Get optimal confidence threshold for the label
                    metric_dict[f"T4MetricV2_label/{label_name}_optimal-confidence_{matching_mode}_{threshold}"] = (
                        ap.optimal_conf
                    )
                    # Optimal recall and precision at the optimal confidence threshold
                    metric_dict[f"T4MetricV2_label/{label_name}_optimal-recall_{matching_mode}_{threshold}"] = (
                        ap.optimal_recall
                    )
                    metric_dict[f"T4MetricV2_label/{label_name}_optimal-precision_{matching_mode}_{threshold}"] = (
                        ap.optimal_precision
                    )

                # Label metadata key
                metric_dict[f"metadata_label/test_{label_name}_num_predictions"] = label_num_preds
                metric_dict[f"metadata_label/test_{label_name}_num_ground_truths"] = label_num_gts

            # Add mAP and mAPH values
            map_key = f"T4MetricV2/mAP_{matching_mode}"
            maph_key = f"T4MetricV2/mAPH_{matching_mode}"
            metric_dict[map_key] = map_instance.map
            metric_dict[maph_key] = map_instance.maph

            total_num_preds = num_preds

        # Selected evaluator
        selected_evaluator = self.evaluators[evaluator_name]

        # Add metadata information
        metric_dict["metadata/experiment_name"] = self.experiment_name
        metric_dict["metadata/experiment_group_name"] = self.experiment_group_name
        metric_dict["metadata/test_timestamp"] = self.test_timestamp
        metric_dict["metadata/test_checkpoint_path"] = self.checkpoint_path
        metric_dict["metadata/test_dataset_name"] = self.dataset_name
        metric_dict["metadata/test_total_num_frames"] = metrics_score.num_frame
        metric_dict["metadata/test_total_num_ground_truths"] = metrics_score.num_ground_truth
        metric_dict["metadata/test_total_num_predictions"] = total_num_preds
        metric_dict["metadata/test_min_range"] = selected_evaluator.min_range
        metric_dict["metadata/test_max_range"] = selected_evaluator.max_range
        metric_dict["metadata/test_range_filter_name"] = selected_evaluator.range_filter_name

        # Add a distribution of the number of frames for each prefix frame
        test_num_frame_distribution = defaultdict(int)
        for used_frame in metrics_score.used_frame:
            test_num_frame_distribution[sample_id_to_prefix_frame_mapping[used_frame]] += 1

        metric_dict["metadata/test_num_frame_distribution"] = test_num_frame_distribution
        return metric_dict

    def _write_aggregated_metrics(
        self, final_metric_dict: dict, aggregated_metric_file_name: str = "aggregated_metrics.json"
    ) -> Dict[str, Any]:
        """
        Writes aggregated metrics to a JSON file with the specified format.

        Args:
            final_metric_dict {evaluator_name: {metric_name: metric_value}}: Dictionary containing processed metrics from the evaluator.
        """
        try:
            # Initialize the structure
            aggregated_metrics = {}
            for evaluator_name in final_metric_dict.keys():
                aggregated_metrics[evaluator_name] = {
                    "metrics": {},
                    "aggregated_metric_label": {},
                    "metadata": {},
                    "metadata_label": {},
                }

            # Gather metrics
            for evaluator_name, metric_dict in final_metric_dict.items():
                # Organize metrics by label
                for key, value in metric_dict.items():
                    if key.startswith("metadata/"):
                        aggregated_metrics[evaluator_name]["metadata"][key] = value
                    elif key.startswith("metadata_label/"):
                        # These are per-label metrics, extract label name and organize
                        # Example: T4MetricV2/car_AP_center_distance_0.5
                        parts = key.split("/")[1].split("_")
                        label_name = parts[1]  # car, truck, etc.
                        if label_name not in aggregated_metrics[evaluator_name]["metadata_label"]:
                            aggregated_metrics[evaluator_name]["metadata_label"][label_name] = {}

                        aggregated_metrics[evaluator_name]["metadata_label"][label_name][key] = value
                    elif key.startswith("T4MetricV2/mAP_") or key.startswith("T4MetricV2/mAPH_"):
                        # These are overall metrics, put them in the metrics section
                        aggregated_metrics[evaluator_name]["metrics"][key] = value
                    else:
                        # These are per-label metrics, extract label name and organize
                        # Example: T4MetricV2/car_AP_center_distance_0.5
                        parts = key.split("/")[1].split("_")
                        label_name = parts[0]  # car, truck, etc.

                        if label_name not in aggregated_metrics[evaluator_name]["aggregated_metric_label"]:
                            aggregated_metrics[evaluator_name]["aggregated_metric_label"][label_name] = {}

                        aggregated_metrics[evaluator_name]["aggregated_metric_label"][label_name][key] = value

            # Write to JSON file
            output_path = self.output_dir / aggregated_metric_file_name
            with open(output_path, "w") as aggregated_file:
                json.dump(aggregated_metrics, aggregated_file, indent=4)

            self.logger.info(f"Aggregated metrics written to: {output_path}")
            return aggregated_metrics

        except Exception as e:
            self.logger.error(f"Failed to write aggregated metrics: {e}")
            raise

    def _write_scene_metrics(self, scenes: dict, evaluator_name: str) -> None:
        """
        Writes scene metrics to a JSON file in nested format.

        Args:
            scenes (dict): Dictionary mapping scene_id to samples, where each sample contains
                          perception frame data.
        """
        try:
            # Initialize scene_metrics structure
            scene_metrics = {
                scene_id: {sample_id: {} for sample_id in samples.keys()} for scene_id, samples in scenes.items()
            }

            # Process all frame results and populate metrics
            self._populate_scene_metrics(scene_metrics, evaluator_name)

            # Write the nested metrics to JSON
            output_path = self.output_dir / evaluator_name / "scene_metrics.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as scene_file:
                json.dump(scene_metrics, scene_file, indent=4)

            self.logger.info(f"Scene metrics written to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to write scene metrics: {e}")
            raise

    def _populate_scene_metrics(self, scene_metrics: dict, evaluator_name: str) -> None:
        """Populate scene metrics with data from frame results.

        Args:
            scene_metrics (dict): The scene metrics structure to populate.
        """
        for frame_info in self.frame_results_with_info[evaluator_name]:
            scene_id = frame_info.scene_id
            sample_id = frame_info.sample_id
            perception_frame_result = frame_info.perception_frame_result

            # Get or create the metrics structure for this frame
            frame_metrics = scene_metrics[scene_id][sample_id].setdefault(
                f"{perception_frame_result.frame_prefix}/{evaluator_name}", {}
            )

            # Process all map instances for a single frame and populate the metrics structure.
            # it iterates through map instances (e.g., center_distance, plane_distance)
            # and processes both AP (Average Precision) and APH (Average Precision with Heading)
            # values for each label and threshold.
            for map_instance in perception_frame_result.metrics_score.mean_ap_values:
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
        location: str = data_sample.get("city", _UNKNOWN)
        vehicle_type: str = data_sample.get("vehicle_type", _UNKNOWN)

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
                frame_id=self.main_evaluator_frame_id,
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
            frame_prefix=location + "/" + vehicle_type,
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
                frame_id=self.main_evaluator_frame_id,
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
