"""
CenterPoint Evaluator for deployment.
"""

import logging
from typing import Dict, List, Mapping, Optional

import numpy as np
from mmengine.config import Config
from typing_extensions import override

from deployment.configs import ComponentsConfig
from deployment.core import (
    BaseEvaluator,
    Detection3DMetricsConfig,
    Detection3DMetricsInterface,
    EvalResultDict,
    InferenceInput,
    ModelSpec,
    TaskProfile,
)
from deployment.core.device import DeviceSpec
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines.base_pipeline import BaseDeploymentPipeline
from deployment.pipelines.factory import PipelineFactory

logger = logging.getLogger(__name__)


class CenterPointEvaluator(BaseEvaluator):
    """Evaluator implementation for CenterPoint 3D detection.

    This builds a task profile (class names, display name) and uses the configured
    `Detection3DMetricsInterface` to compute metrics from pipeline outputs.

    Args:
        model_cfg: Model configuration with class_names
        metrics_config: Configuration for 3D detection metrics
        components_cfg: Unified components configuration (ComponentsConfig).
                       Used to get output names from pts_backbone_neck_head.io.outputs
    """

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: Detection3DMetricsConfig,
        components_cfg: ComponentsConfig,
    ):
        """Initialize CenterPoint evaluator with model config, metrics config, and components config.

        Args:
            model_cfg: Model configuration; must have class_names.
            metrics_config: Configuration for 3D detection metrics (e.g. T4MetricV2).
            components_cfg: Unified components config; used for output names of pts_backbone_neck_head.

        Raises:
            ValueError: If model_cfg does not have class_names.
        """
        if hasattr(model_cfg, "class_names"):
            class_names = model_cfg.class_names
        else:
            raise ValueError("class_names must be provided via model_cfg.class_names.")

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
        """Set the evaluator's model config to the given ONNX-compatible config.

        Args:
            model_cfg: ONNX-compatible model configuration (e.g. from build_centerpoint_onnx_model).
        """
        self.model_cfg = model_cfg

    # VerificationMixin
    @override
    def _get_output_names(self) -> Optional[List[str]]:
        """Get head output names from components config."""
        return [out.name for out in self._components_cfg.get_component("pts_backbone_neck_head").io.outputs]

    @override
    def _create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseDeploymentPipeline:
        """Create a CenterPoint deployment pipeline for the given backend and device.

        Args:
            model_spec: Model specification (backend, device, path).
            device: Target device for the pipeline.

        Returns:
            CenterPoint pipeline instance (PyTorch, ONNX, or TensorRT).
        """
        return PipelineFactory.create(
            project_name="centerpoint",
            model_spec=model_spec,
            pytorch_model=self.pytorch_model,
            device=device,
            components_cfg=self._components_cfg,
        )

    @override
    def _prepare_input(
        self,
        sample: Mapping[str, object],
        data_loader: BaseDataLoader,
        device: DeviceSpec,
    ) -> InferenceInput:
        """Build InferenceInput from sample (points + metainfo).

        Args:
            sample: Dict with 'points' and 'metainfo'.
            data_loader: Unused; kept for interface compatibility.
            device: Unused; kept for interface compatibility.

        Returns:
            InferenceInput with data=points and metadata=metainfo.

        Raises:
            ValueError: If 'points' is missing from sample.
            KeyError: If 'metainfo' is missing from sample.
        """
        if "points" not in sample:
            raise ValueError(f"Expected 'points' in sample. Got keys: {list(sample.keys())}")
        if "metainfo" not in sample:
            raise KeyError("Sample must contain 'metainfo' for CenterPoint postprocess.")
        points = sample["points"]
        metadata = sample["metainfo"]
        return InferenceInput(data=points, metadata=metadata)

    @override
    def _parse_predictions(self, pipeline_output: object) -> List[Dict]:
        """Return pipeline output as a list of prediction dicts (or empty list if not a list).

        Args:
            pipeline_output: Raw output from the deployment pipeline.

        Returns:
            List of prediction dicts, or empty list if pipeline_output is not a list.
        """
        return pipeline_output if isinstance(pipeline_output, list) else []

    @override
    def _parse_ground_truths(self, gt_data: Mapping[str, object]) -> List[Dict]:
        """Convert gt_bboxes_3d and gt_labels_3d into list of dicts with bbox_3d and label.

        Args:
            gt_data: Dict with 'gt_bboxes_3d' and 'gt_labels_3d'.

        Returns:
            List of {"bbox_3d": [...], "label": int}.

        Raises:
            KeyError: If gt_bboxes_3d or gt_labels_3d is missing.
        """
        ground_truths = []

        if "gt_bboxes_3d" not in gt_data:
            raise KeyError("gt_bboxes_3d not found in ground truth data.")
        if "gt_labels_3d" not in gt_data:
            raise KeyError("gt_labels_3d not found in ground truth data.")

        gt_bboxes_3d = gt_data["gt_bboxes_3d"]
        gt_labels_3d = gt_data["gt_labels_3d"]

        gt_bboxes_3d = np.asarray(gt_bboxes_3d, dtype=np.float32).reshape(
            -1, np.asarray(gt_bboxes_3d).shape[-1] if np.asarray(gt_bboxes_3d).ndim > 1 else 7
        )
        gt_labels_3d = np.asarray(gt_labels_3d, dtype=np.int64).reshape(-1)

        for i in range(len(gt_bboxes_3d)):
            ground_truths.append({"bbox_3d": gt_bboxes_3d[i].tolist(), "label": int(gt_labels_3d[i])})

        return ground_truths

    @override
    def _add_to_interface(self, predictions: List[Dict], ground_truths: List[Dict]) -> None:
        """Add one frame of predictions and ground truths to the metrics interface.

        Args:
            predictions: List of prediction dicts (bbox_3d, score, label).
            ground_truths: List of ground truth dicts (bbox_3d, label).
        """
        self.metrics_interface.add_frame(predictions, ground_truths)

    @override
    def _build_results(
        self,
        latencies: List[float],
        latency_breakdowns: List[Dict[str, float]],
        num_samples: int,
    ) -> EvalResultDict:
        """Build evaluation result dict with mAP/mAPH, per-class AP, latency, and optional breakdown.

        Args:
            latencies: Per-sample inference latencies (ms).
            latency_breakdowns: Per-sample stage-wise latencies (optional).
            num_samples: Number of evaluated samples.

        Returns:
            EvalResultDict with mAP_by_mode, mAPH_by_mode, per_class_ap_by_mode,
            detailed_metrics, latency stats, num_samples, and optionally latency_breakdown.

        Raises:
            KeyError: If metrics summary is missing required keys.
        """
        latency_stats = self.compute_latency_stats(latencies)

        map_results = self.metrics_interface.compute_metrics()
        summary = self.metrics_interface.summary
        summary_dict = summary.to_dict()
        required_summary_keys = ("mAP_by_mode", "mAPH_by_mode", "per_class_ap_by_mode")
        missing = [k for k in required_summary_keys if k not in summary_dict]
        if missing:
            raise KeyError(f"Missing required metrics summary keys: {missing}")

        result: EvalResultDict = {
            "mAP_by_mode": summary_dict["mAP_by_mode"],
            "mAPH_by_mode": summary_dict["mAPH_by_mode"],
            "per_class_ap_by_mode": summary_dict["per_class_ap_by_mode"],
            "detailed_metrics": map_results,
            "latency": latency_stats,
            "num_samples": num_samples,
        }

        if latency_breakdowns:
            result["latency_breakdown"] = self._compute_latency_breakdown(latency_breakdowns)

        return result

    @override
    def print_results(self, results: EvalResultDict) -> None:
        """Print evaluation results including metrics, latency, and breakdown.

        Args:
            results: EvalResultDict from _build_results (mAP, latency, num_samples, etc.).

        Raises:
            ValueError: If metrics report or latency is missing from results.
        """
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

        print(f"\nTotal Samples: {results['num_samples']}")
