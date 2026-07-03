"""
CenterPoint-specific deployment runner.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from mmengine.config import Config

from deployment.config.base import BaseDeploymentConfig
from deployment.evaluation.backend_executor import BackendExecutor
from deployment.export.contexts import ExportContext
from deployment.export.pipelines.onnx_export_pipeline import OnnxExportPipeline
from deployment.export.pipelines.tensorrt_export_pipeline import TensorRTExportPipeline
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.projects.centerpoint.contexts import CenterPointExportContext
from deployment.projects.centerpoint.evaluation.evaluator import CenterPointEvaluator
from deployment.projects.centerpoint.export.component_builder import CenterPointComponentBuilder
from deployment.projects.centerpoint.export.sample_extractor import CenterPointSampleExtractor
from deployment.projects.centerpoint.io.model_loader import build_centerpoint_onnx_model
from deployment.runtime.runner import BaseDeploymentRunner

logger = logging.getLogger(__name__)


class CenterPointDeploymentRunner(BaseDeploymentRunner):
    """CenterPoint deployment runner.

    Implements project-specific model loading and wiring to export pipelines,
    while reusing the project-agnostic orchestration in `BaseDeploymentRunner`.

    Attributes:
        model_cfg: Training MMEngine config (from checkpoint experiment file); not replaced after load.
        evaluator: CenterPoint evaluator instance.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: CenterPointEvaluator,
        executor: BackendExecutor,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        """Initialize CenterPoint deployment runner.

        Args:
            data_loader: Data loader for loading samples.
            evaluator: Evaluator for computing metrics.
            executor: Backend execution primitives shared with the evaluator/verification runner.
            config: Deployment configuration.
            model_cfg: MMEngine model configuration.
            onnx_pipeline: Optional custom ONNX export pipeline.
            tensorrt_pipeline: Optional custom TensorRT export pipeline.
        """

        # CenterPoint must be split into two ONNX components (voxel encoder and
        # backbone/neck/head), so it drives the shared OnnxExportPipeline with a
        # project-specific sample extractor + component builder instead of the
        # whole-model default. IdentityWrapper (the pipeline default) is used as
        # the components need no ONNX output reshaping.
        if onnx_pipeline is None:
            onnx_pipeline = OnnxExportPipeline(
                sample_extractor=CenterPointSampleExtractor(),
                component_builder=CenterPointComponentBuilder(components_cfg=config.components_cfg),
            )
        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            executor=executor,
            config=config,
            model_cfg=model_cfg,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> torch.nn.Module:
        """Load and return the PyTorch model for export.

        Args:
            checkpoint_path: Path to the checkpoint file.
            context: Export context with additional parameters.

        Returns:
            Loaded PyTorch model.
        """
        rot_y_axis_reference = self._extract_rot_y_axis_reference(context)
        logger.info("Export option rot_y_axis_reference = %s", rot_y_axis_reference)

        model, _ = build_centerpoint_onnx_model(
            base_model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device=DeviceSpec.from_value("cpu"),
            rot_y_axis_reference=rot_y_axis_reference,
        )
        return model

    def _extract_rot_y_axis_reference(self, context: ExportContext) -> bool:
        """Extract rot_y_axis_reference from the export context.

        Args:
            context: Export context; must be a ``CenterPointExportContext``.

        Returns:
            Boolean value for rot_y_axis_reference.
        """
        if not isinstance(context, CenterPointExportContext):
            raise TypeError(f"CenterPoint export requires a CenterPointExportContext, got {type(context).__name__}.")
        return context.rot_y_axis_reference
