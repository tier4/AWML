"""
CenterPoint-specific deployment runner.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from mmengine.config import Config

from deployment.core import BaseDeploymentConfig
from deployment.core.contexts import CenterPointExportContext, ExportContext
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import IdentityWrapper
from deployment.exporters.export_pipelines.base import OnnxExportPipeline, TensorRTExportPipeline
from deployment.projects.centerpoint.evaluator import CenterPointEvaluator
from deployment.projects.centerpoint.export.component_extractor import CenterPointComponentExtractor
from deployment.projects.centerpoint.export.onnx_export_pipeline import CenterPointONNXExportPipeline
from deployment.projects.centerpoint.export.tensorrt_export_pipeline import CenterPointTensorRTExportPipeline
from deployment.projects.centerpoint.model_loader import build_centerpoint_onnx_model
from deployment.runtime.runner import BaseDeploymentRunner


class CenterPointDeploymentRunner(BaseDeploymentRunner):
    """CenterPoint deployment runner.

    Implements project-specific model loading and wiring to export pipelines,
    while reusing the project-agnostic orchestration in `BaseDeploymentRunner`.

    Attributes:
        model_cfg: MMEngine model configuration.
        evaluator: CenterPoint evaluator instance.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: CenterPointEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        onnx_pipeline: Optional[OnnxExportPipeline] = None,
        tensorrt_pipeline: Optional[TensorRTExportPipeline] = None,
    ) -> None:
        """Initialize CenterPoint deployment runner.

        Args:
            data_loader: Data loader for loading samples.
            evaluator: Evaluator for computing metrics.
            config: Deployment configuration.
            model_cfg: MMEngine model configuration.
            logger: Logger instance.
            onnx_pipeline: Optional custom ONNX export pipeline.
            tensorrt_pipeline: Optional custom TensorRT export pipeline.
        """
        component_extractor = CenterPointComponentExtractor(config=config, logger=logger)

        super().__init__(
            data_loader=data_loader,
            evaluator=evaluator,
            config=config,
            model_cfg=model_cfg,
            logger=logger,
            onnx_wrapper_cls=IdentityWrapper,
            onnx_pipeline=onnx_pipeline,
            tensorrt_pipeline=tensorrt_pipeline,
        )

        if self._onnx_pipeline is None:
            self._onnx_pipeline = CenterPointONNXExportPipeline(
                exporter_factory=ExporterFactory,
                component_extractor=component_extractor,
                logger=self.logger,
            )

        if self._tensorrt_pipeline is None:
            self._tensorrt_pipeline = CenterPointTensorRTExportPipeline(
                exporter_factory=ExporterFactory,
                logger=self.logger,
            )

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> torch.nn.Module:
        """Load PyTorch model for export.

        Args:
            checkpoint_path: Path to the checkpoint file.
            context: Export context with additional parameters.

        Returns:
            Loaded PyTorch model.
        """
        rot_y_axis_reference = self._extract_rot_y_axis_reference(context)

        model, onnx_cfg = build_centerpoint_onnx_model(
            base_model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device="cpu",
            rot_y_axis_reference=rot_y_axis_reference,
        )

        # Update model_cfg with ONNX-compatible version
        self.model_cfg = onnx_cfg

        # Notify evaluator of model availability
        self._setup_evaluator(model, onnx_cfg)

        return model

    def _extract_rot_y_axis_reference(self, context: ExportContext) -> bool:
        """Extract rot_y_axis_reference from export context.

        Args:
            context: Export context (typed or dict-like).

        Returns:
            Boolean value for rot_y_axis_reference.
        """
        if isinstance(context, CenterPointExportContext):
            return context.rot_y_axis_reference
        return context.get("rot_y_axis_reference", False)

    def _setup_evaluator(self, model: torch.nn.Module, onnx_cfg: Config) -> None:
        """Setup evaluator with loaded model and config.

        This method updates the evaluator with the PyTorch model and
        ONNX-compatible configuration needed for evaluation.

        Args:
            model: Loaded PyTorch model.
            onnx_cfg: ONNX-compatible model configuration.
        """
        try:
            self.evaluator.set_onnx_config(onnx_cfg)
            self.logger.info("Updated evaluator with ONNX-compatible config")
        except Exception as e:
            self.logger.error(f"Failed to update evaluator config: {e}")
            raise

        try:
            self.evaluator.set_pytorch_model(model)
            self.logger.info("Updated evaluator with PyTorch model")
        except Exception as e:
            self.logger.error(f"Failed to set PyTorch model on evaluator: {e}")
            raise
