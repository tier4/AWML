"""
CenterPoint-specific deployment runner.
"""

from __future__ import annotations

import logging
from typing import Any

from deployment.core.contexts import CenterPointExportContext, ExportContext
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import IdentityWrapper
from deployment.projects.centerpoint.export.component_extractor import CenterPointComponentExtractor
from deployment.projects.centerpoint.export.onnx_export_pipeline import CenterPointONNXExportPipeline
from deployment.projects.centerpoint.export.tensorrt_export_pipeline import CenterPointTensorRTExportPipeline
from deployment.projects.centerpoint.model_loader import build_centerpoint_onnx_model
from deployment.runtime.runner import BaseDeploymentRunner


class CenterPointDeploymentRunner(BaseDeploymentRunner):
    """CenterPoint deployment runner.

    Implements project-specific model loading and wiring to export pipelines,
    while reusing the project-agnostic orchestration in `BaseDeploymentRunner`.
    """

    def __init__(
        self,
        data_loader,
        evaluator,
        config,
        model_cfg,
        logger: logging.Logger,
        onnx_pipeline=None,
        tensorrt_pipeline=None,
    ):
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
                config=self.config,
                logger=self.logger,
            )

        if self._tensorrt_pipeline is None:
            self._tensorrt_pipeline = CenterPointTensorRTExportPipeline(
                exporter_factory=ExporterFactory,
                config=self.config,
                logger=self.logger,
            )

    def load_pytorch_model(self, checkpoint_path: str, context: ExportContext) -> Any:
        rot_y_axis_reference: bool = False
        if isinstance(context, CenterPointExportContext):
            rot_y_axis_reference = context.rot_y_axis_reference
        else:
            rot_y_axis_reference = context.get("rot_y_axis_reference", False)

        model, onnx_cfg = build_centerpoint_onnx_model(
            base_model_cfg=self.model_cfg,
            checkpoint_path=checkpoint_path,
            device="cpu",
            rot_y_axis_reference=rot_y_axis_reference,
        )

        self.model_cfg = onnx_cfg
        self._inject_model_to_evaluator(model, onnx_cfg)
        return model

    def _inject_model_to_evaluator(self, model: Any, onnx_cfg: Any) -> None:
        try:
            self.evaluator.set_onnx_config(onnx_cfg)
            self.logger.info("Injected ONNX-compatible config to evaluator")
        except Exception as e:
            self.logger.error(f"Failed to inject ONNX config: {e}")
            raise

        try:
            self.evaluator.set_pytorch_model(model)
            self.logger.info("Injected PyTorch model to evaluator")
        except Exception as e:
            self.logger.error(f"Failed to inject PyTorch model: {e}")
            raise
