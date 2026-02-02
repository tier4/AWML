"""
CenterPoint Evaluator for deployment.
"""

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from mmengine.config import Config

from deployment.core import (
    BaseEvaluator,
    Detection3DMetricsConfig,
    Detection3DMetricsInterface,
    EvalResultDict,
    InferenceInput,
    ModelSpec,
    TaskProfile,
)
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines.factory import PipelineFactory

logger = logging.getLogger(__name__)


class CenterPointEvaluator(BaseEvaluator):
    """Evaluator implementation for CenterPoint 3D detection.

    This builds a task profile (class names, display name) and uses the configured
    `Detection3DMetricsInterface` to compute metrics from pipeline outputs.

    Args:
        model_cfg: Model configuration with class_names
        metrics_config: Configuration for 3D detection metrics
        components_cfg: Optional unified components configuration dict.
                       Used to get output names from components.backbone_head.io.outputs
    """

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: Detection3DMetricsConfig,
        components_cfg: Optional[Mapping[str, Any]] = None,
    ):
        if hasattr(model_cfg, "class_names"):
            class_names = model_cfg.class_names
        else:
            raise ValueError("class_names must be provided via model_cfg.class_names.")

        if components_cfg is None:
            components_cfg = {}
        if not isinstance(components_cfg, Mapping):
            raise TypeError(f"components_cfg must be a mapping, got {type(components_cfg).__name__}")
        self._components_cfg = components_cfg

        task_profile = TaskProfile(
            task_name="centerpoint_3d_detection",
            display_name="CenterPoint 3D Object Detection",
            class_names=tuple(class_names),
            num_classes=len(class_names),
        )

        metrics_interface = Detection3DMetricsInterface(metrics_config)

        super().__init__(
            metrics_interface=metrics_interface,
            task_profile=task_profile,
            model_cfg=model_cfg,
        )

    def set_onnx_config(self, model_cfg: Config) -> None:
        self.model_cfg = model_cfg

    def _get_output_names(self) -> List[str]:
        """Get head output names from components config."""
        backbone_head_cfg = self._components_cfg.get("backbone_head", {})
        io_cfg = backbone_head_cfg.get("io", {})
        outputs = io_cfg.get("outputs", [])

        if not outputs:
            raise ValueError(
                "Output names must be provided via components_cfg.backbone_head.io.outputs. "
                "No fallback values are allowed in deployment framework."
            )

        output_names = [out.get("name") for out in outputs if out.get("name")]
        if not output_names:
            raise ValueError(
                "Output names must be provided via components_cfg.backbone_head.io.outputs. "
                "Each output must have a 'name' field."
            )

        return output_names

    def _create_pipeline(self, model_spec: ModelSpec, device: str) -> Any:
        return PipelineFactory.create(
            project_name="centerpoint",
            model_spec=model_spec,
            pytorch_model=self.pytorch_model,
            device=device,
            components_cfg=self._components_cfg,
        )

    def _prepare_input(
        self,
        sample: Dict[str, Any],
        data_loader: BaseDataLoader,
        device: str,
    ) -> InferenceInput:
        if "points" in sample:
            points = sample["points"]
            metadata = sample.get("metainfo", {})
        else:
            raise ValueError(f"Expected 'points' in sample. Got keys: {list(sample.keys())}")
        return InferenceInput(data=points, metadata=metadata)

    def _parse_predictions(self, pipeline_output: Any) -> List[Dict]:
        return pipeline_output if isinstance(pipeline_output, list) else []

    def _parse_ground_truths(self, gt_data: Dict[str, Any]) -> List[Dict]:
        ground_truths = []

        if "gt_bboxes_3d" in gt_data and "gt_labels_3d" in gt_data:
            gt_bboxes_3d = gt_data["gt_bboxes_3d"]
            gt_labels_3d = gt_data["gt_labels_3d"]

            gt_bboxes_3d = np.asarray(gt_bboxes_3d, dtype=np.float32).reshape(
                -1, np.asarray(gt_bboxes_3d).shape[-1] if np.asarray(gt_bboxes_3d).ndim > 1 else 7
            )
            gt_labels_3d = np.asarray(gt_labels_3d, dtype=np.int64).reshape(-1)

            for i in range(len(gt_bboxes_3d)):
                ground_truths.append({"bbox_3d": gt_bboxes_3d[i].tolist(), "label": int(gt_labels_3d[i])})

        return ground_truths

    def _add_to_interface(self, predictions: List[Dict], ground_truths: List[Dict]) -> None:
        self.metrics_interface.add_frame(predictions, ground_truths)

    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        latency_stats = self.compute_latency_stats(latencies)

        map_results = self.metrics_interface.compute_metrics()
        summary = self.metrics_interface.summary
        summary_dict = summary.to_dict() if hasattr(summary, "to_dict") else summary

        result: EvalResultDict = {
            "mAP_by_mode": summary_dict.get("mAP_by_mode", {}),
            "mAPH_by_mode": summary_dict.get("mAPH_by_mode", {}),
            "per_class_ap_by_mode": summary_dict.get("per_class_ap_by_mode", {}),
            "detailed_metrics": map_results,
            "latency": latency_stats,  # Store LatencyStats directly
            "num_samples": num_samples,
        }

        if latency_breakdowns:
            result["latency_breakdown"] = self._compute_latency_breakdown(latency_breakdowns)

        return result

    def print_results(self, results: EvalResultDict) -> None:
        """Print evaluation results including metrics, latency, and breakdown."""
        # Print metrics report
        metrics_report = self.metrics_interface.format_metrics_report()
        if not metrics_report:
            raise ValueError(
                "Metrics report is empty. Ensure that frames have been added and metrics have been computed."
            )
        print(metrics_report)

        # Print latency statistics
        if "latency" not in results:
            raise ValueError(
                "Latency statistics not found in results. Ensure that evaluation has been run with latency tracking."
            )
        latency_stats = results["latency"]
        latency_dict = latency_stats.to_dict()
        print("\nLatency Statistics:")
        print(f"  Mean:   {latency_dict['mean_ms']:.2f} ms")
        print(f"  Std:    {latency_dict['std_ms']:.2f} ms")
        print(f"  Min:    {latency_dict['min_ms']:.2f} ms")
        print(f"  Max:    {latency_dict['max_ms']:.2f} ms")
        print(f"  Median: {latency_dict['median_ms']:.2f} ms")

        # Print stage-wise latency breakdown
        if "latency_breakdown" in results:
            breakdown = results["latency_breakdown"]
            breakdown_dict = breakdown.to_dict() if hasattr(breakdown, "to_dict") else breakdown

            if breakdown_dict:
                print("\nStage-wise Latency Breakdown:")
                top_level_stages = {"preprocessing_ms", "model_ms", "postprocessing_ms"}
                for stage, stats in breakdown_dict.items():
                    stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else stats
                    stage_name = stage.replace("_ms", "").replace("_", " ").title()

                    if stage in top_level_stages:
                        print(f"  {stage_name:18s}: {stats_dict['mean_ms']:.2f} ± {stats_dict['std_ms']:.2f} ms")
                    else:
                        print(f"    {stage_name:16s}: {stats_dict['mean_ms']:.2f} ± {stats_dict['std_ms']:.2f} ms")

        print(f"\nTotal Samples: {results.get('num_samples', 0)}")


"""
CenterPoint Evaluator for deployment.
"""

import logging
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from mmengine.config import Config

from deployment.core import (
    BaseEvaluator,
    Detection3DMetricsConfig,
    Detection3DMetricsInterface,
    EvalResultDict,
    InferenceInput,
    ModelSpec,
    TaskProfile,
)
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines.factory import PipelineFactory

logger = logging.getLogger(__name__)


class CenterPointEvaluator(BaseEvaluator):
    """Evaluator implementation for CenterPoint 3D detection.

    This builds a task profile (class names, display name) and uses the configured
    `Detection3DMetricsInterface` to compute metrics from pipeline outputs.

    Args:
        model_cfg: Model configuration with class_names
        metrics_config: Configuration for 3D detection metrics
        components_cfg: Optional unified components configuration dict.
                       Used to get output names from components.backbone_head.io.outputs
    """

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: Detection3DMetricsConfig,
        components_cfg: Optional[Mapping[str, Any]] = None,
    ):
        if hasattr(model_cfg, "class_names"):
            class_names = model_cfg.class_names
        else:
            raise ValueError("class_names must be provided via model_cfg.class_names.")

        if components_cfg is None:
            components_cfg = {}
        if not isinstance(components_cfg, Mapping):
            raise TypeError(f"components_cfg must be a mapping, got {type(components_cfg).__name__}")
        self._components_cfg = components_cfg

        task_profile = TaskProfile(
            task_name="centerpoint_3d_detection",
            display_name="CenterPoint 3D Object Detection",
            class_names=tuple(class_names),
            num_classes=len(class_names),
        )

        metrics_interface = Detection3DMetricsInterface(metrics_config)

        super().__init__(
            metrics_interface=metrics_interface,
            task_profile=task_profile,
            model_cfg=model_cfg,
        )

    def set_onnx_config(self, model_cfg: Config) -> None:
        self.model_cfg = model_cfg

    def _get_output_names(self) -> List[str]:
        """Get head output names from components config."""
        backbone_head_cfg = self._components_cfg.get("backbone_head", {})
        io_cfg = backbone_head_cfg.get("io", {})
        outputs = io_cfg.get("outputs", [])

        if not outputs:
            raise ValueError(
                "Output names must be provided via components_cfg.backbone_head.io.outputs. "
                "No fallback values are allowed in deployment framework."
            )

        output_names = [out.get("name") for out in outputs if out.get("name")]
        if not output_names:
            raise ValueError(
                "Output names must be provided via components_cfg.backbone_head.io.outputs. "
                "Each output must have a 'name' field."
            )

        return output_names

    def _create_pipeline(self, model_spec: ModelSpec, device: str) -> Any:
        return PipelineFactory.create(
            project_name="centerpoint",
            model_spec=model_spec,
            pytorch_model=self.pytorch_model,
            device=device,
            components_cfg=self._components_cfg,
        )

    def _prepare_input(
        self,
        sample: Dict[str, Any],
        data_loader: BaseDataLoader,
        device: str,
    ) -> InferenceInput:
        if "points" in sample:
            points = sample["points"]
            metadata = sample.get("metainfo", {})
        else:
            raise ValueError(f"Expected 'points' in sample. Got keys: {list(sample.keys())}")
        return InferenceInput(data=points, metadata=metadata)

    def _parse_predictions(self, pipeline_output: Any) -> List[Dict]:
        return pipeline_output if isinstance(pipeline_output, list) else []

    def _parse_ground_truths(self, gt_data: Dict[str, Any]) -> List[Dict]:
        ground_truths = []

        if "gt_bboxes_3d" in gt_data and "gt_labels_3d" in gt_data:
            gt_bboxes_3d = gt_data["gt_bboxes_3d"]
            gt_labels_3d = gt_data["gt_labels_3d"]

            gt_bboxes_3d = np.asarray(gt_bboxes_3d, dtype=np.float32).reshape(
                -1, np.asarray(gt_bboxes_3d).shape[-1] if np.asarray(gt_bboxes_3d).ndim > 1 else 7
            )
            gt_labels_3d = np.asarray(gt_labels_3d, dtype=np.int64).reshape(-1)

            for i in range(len(gt_bboxes_3d)):
                ground_truths.append({"bbox_3d": gt_bboxes_3d[i].tolist(), "label": int(gt_labels_3d[i])})

        return ground_truths

    def _add_to_interface(self, predictions: List[Dict], ground_truths: List[Dict]) -> None:
        self.metrics_interface.add_frame(predictions, ground_truths)

    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        latency_stats = self.compute_latency_stats(latencies)

        map_results = self.metrics_interface.compute_metrics()
        summary = self.metrics_interface.summary
        summary_dict = summary.to_dict() if hasattr(summary, "to_dict") else summary

        result: EvalResultDict = {
            "mAP_by_mode": summary_dict.get("mAP_by_mode", {}),
            "mAPH_by_mode": summary_dict.get("mAPH_by_mode", {}),
            "per_class_ap_by_mode": summary_dict.get("per_class_ap_by_mode", {}),
            "detailed_metrics": map_results,
            "latency": latency_stats,  # Store LatencyStats directly
            "num_samples": num_samples,
        }

        if latency_breakdowns:
            result["latency_breakdown"] = self._compute_latency_breakdown(latency_breakdowns)

        return result

    def print_results(self, results: EvalResultDict) -> None:
        """Print evaluation results including metrics, latency, and breakdown."""
        # Print metrics report
        metrics_report = self.metrics_interface.format_metrics_report()
        if not metrics_report:
            raise ValueError(
                "Metrics report is empty. Ensure that frames have been added and metrics have been computed."
            )
        print(metrics_report)

        # Print latency statistics
        if "latency" not in results:
            raise ValueError(
                "Latency statistics not found in results. Ensure that evaluation has been run with latency tracking."
            )
        latency_stats = results["latency"]
        latency_dict = latency_stats.to_dict()
        print("\nLatency Statistics:")
        print(f"  Mean:   {latency_dict['mean_ms']:.2f} ms")
        print(f"  Std:    {latency_dict['std_ms']:.2f} ms")
        print(f"  Min:    {latency_dict['min_ms']:.2f} ms")
        print(f"  Max:    {latency_dict['max_ms']:.2f} ms")
        print(f"  Median: {latency_dict['median_ms']:.2f} ms")

        # Print stage-wise latency breakdown
        if "latency_breakdown" in results:
            breakdown = results["latency_breakdown"]
            breakdown_dict = breakdown.to_dict() if hasattr(breakdown, "to_dict") else breakdown

            if breakdown_dict:
                print("\nStage-wise Latency Breakdown:")
                top_level_stages = {"preprocessing_ms", "model_ms", "postprocessing_ms"}
                for stage, stats in breakdown_dict.items():
                    stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else stats
                    stage_name = stage.replace("_ms", "").replace("_", " ").title()

                    if stage in top_level_stages:
                        print(f"  {stage_name:18s}: {stats_dict['mean_ms']:.2f} ± {stats_dict['std_ms']:.2f} ms")
                    else:
                        print(f"    {stage_name:16s}: {stats_dict['mean_ms']:.2f} ± {stats_dict['std_ms']:.2f} ms")

        print(f"\nTotal Samples: {results.get('num_samples', 0)}")
