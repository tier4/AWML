"""
CenterPoint Evaluator for deployment.
"""

import logging
from typing import Any, Dict, List, Tuple

from mmengine.config import Config

from deployment.core import (
    BaseEvaluator,
    Detection3DMetricsConfig,
    Detection3DMetricsInterface,
    EvalResultDict,
    ModelSpec,
    TaskProfile,
)
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.pipelines.factory import PipelineFactory
from deployment.projects.centerpoint.config.deploy_config import model_io

logger = logging.getLogger(__name__)


class CenterPointEvaluator(BaseEvaluator):
    """Evaluator implementation for CenterPoint 3D detection.

    This builds a task profile (class names, display name) and uses the configured
    `Detection3DMetricsInterface` to compute metrics from pipeline outputs.
    """

    def __init__(
        self,
        model_cfg: Config,
        metrics_config: Detection3DMetricsConfig,
    ):
        if hasattr(model_cfg, "class_names"):
            class_names = model_cfg.class_names
        else:
            raise ValueError("class_names must be provided via model_cfg.class_names.")

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
        return list(model_io["head_output_names"])

    def _create_pipeline(self, model_spec: ModelSpec, device: str) -> Any:
        return PipelineFactory.create(
            project_name="centerpoint",
            model_spec=model_spec,
            pytorch_model=self.pytorch_model,
            device=device,
        )

    def _prepare_input(
        self,
        sample: Dict[str, Any],
        data_loader: BaseDataLoader,
        device: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        if "points" in sample:
            points = sample["points"]
        else:
            input_data = data_loader.preprocess(sample)
            points = input_data.get("points", input_data)

        metadata = sample.get("metainfo", {})
        return points, metadata

    def _parse_predictions(self, pipeline_output: Any) -> List[Dict]:
        return pipeline_output if isinstance(pipeline_output, list) else []

    def _parse_ground_truths(self, gt_data: Dict[str, Any]) -> List[Dict]:
        ground_truths = []

        if "gt_bboxes_3d" in gt_data and "gt_labels_3d" in gt_data:
            gt_bboxes_3d = gt_data["gt_bboxes_3d"]
            gt_labels_3d = gt_data["gt_labels_3d"]

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
        latency_payload = latency_stats.to_dict()

        if latency_breakdowns:
            latency_payload["latency_breakdown"] = self._compute_latency_breakdown(latency_breakdowns).to_dict()

        map_results = self.metrics_interface.compute_metrics()
        summary = self.metrics_interface.get_summary()
        summary_dict = summary.to_dict() if hasattr(summary, "to_dict") else summary

        return {
            "mAP": summary_dict.get("mAP", 0.0),
            "mAPH": summary_dict.get("mAPH", 0.0),
            "per_class_ap": summary_dict.get("per_class_ap", {}),
            "detailed_metrics": map_results,
            "latency": latency_payload,
            "num_samples": num_samples,
        }

    def print_results(self, results: EvalResultDict) -> None:
        print("\n" + "=" * 80)
        print(f"{self.task_profile.display_name} - Evaluation Results")
        print("(Using autoware_perception_evaluation for consistent metrics)")
        print("=" * 80)

        print("\nDetection Metrics:")
        print(f"  mAP: {results.get('mAP', 0.0):.4f}")
        print(f"  mAPH: {results.get('mAPH', 0.0):.4f}")

        if "per_class_ap" in results:
            print("\nPer-Class AP:")
            for class_id, ap in results["per_class_ap"].items():
                class_name = (
                    class_id
                    if isinstance(class_id, str)
                    else (self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}")
                )
                print(f"  {class_name:<12}: {ap:.4f}")

        if "latency" in results:
            latency = results["latency"]
            print("\nLatency Statistics:")
            print(f"  Mean:   {latency['mean_ms']:.2f} ms")
            print(f"  Std:    {latency['std_ms']:.2f} ms")
            print(f"  Min:    {latency['min_ms']:.2f} ms")
            print(f"  Max:    {latency['max_ms']:.2f} ms")
            print(f"  Median: {latency['median_ms']:.2f} ms")

            if "latency_breakdown" in latency:
                breakdown = latency["latency_breakdown"]
                print("\nStage-wise Latency Breakdown:")
                model_substages = {"voxel_encoder_ms", "middle_encoder_ms", "backbone_head_ms"}
                for stage, stats in breakdown.items():
                    stage_name = stage.replace("_ms", "").replace("_", " ").title()
                    if stage in model_substages:
                        print(f"    {stage_name:16s}: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
                    else:
                        print(f"  {stage_name:18s}: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")

        print(f"\nTotal Samples: {results.get('num_samples', 0)}")
        print("=" * 80)
