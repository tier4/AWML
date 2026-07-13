"""
CenterPoint backend executor.

Implements the task-specific backend execution primitives (pipeline creation and
input preparation) for CenterPoint, shared by the evaluator and the verification
runner via `~deployment.execution.backend_executor.BackendExecutor`.
"""

import logging
from typing import List, Optional

from typing_extensions import override

from deployment.config.enums import Backend
from deployment.config.schema import ComponentsConfig
from deployment.execution.point_cloud_backend_executor import PointCloudBackendExecutor
from deployment.inference.base_inference_pipeline import BaseInferencePipeline
from deployment.primitives.device import DeviceSpec
from deployment.primitives.evaluator_types import ModelSpec
from deployment.projects.centerpoint.inference.onnx_inference_pipeline import CenterPointONNXInferencePipeline
from deployment.projects.centerpoint.inference.pytorch_inference_pipeline import CenterPointPyTorchInferencePipeline
from deployment.projects.centerpoint.inference.tensorrt_inference_pipeline import CenterPointTensorRTInferencePipeline

logger = logging.getLogger(__name__)


class CenterPointExecutor(PointCloudBackendExecutor):
    """Backend execution primitives for CenterPoint (pipeline creation, input prep).

    Args:
        components_cfg: Unified components configuration, forwarded to the pipeline
            registry when constructing backend pipelines.
    """

    def __init__(self, components_cfg: ComponentsConfig) -> None:
        super().__init__()
        self._components_cfg = components_cfg

    @override
    def get_output_names(self) -> Optional[List[str]]:
        """Return the head output names from the components config for verification logging."""
        return [out.name for out in self._components_cfg.get_component("pts_backbone_neck_head").io.outputs]

    @override
    def create_pipeline(self, model_spec: ModelSpec, device: DeviceSpec) -> BaseInferencePipeline:
        """Create a CenterPoint inference pipeline for the given backend and device.

        Args:
            model_spec: Model specification (backend, device, path).
            device: Target device for the pipeline.

        Returns:
            CenterPoint pipeline instance (PyTorch, ONNX, or TensorRT).

        Raises:
            ValueError: If ``model_spec.backend`` is not a supported backend.
        """
        backend = model_spec.backend
        self._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info("Creating CenterPoint PyTorch pipeline on %s", device)
            return CenterPointPyTorchInferencePipeline(self.pytorch_model, device=device)

        if backend is Backend.ONNX:
            logger.info("Creating CenterPoint ONNX pipeline from %s on %s", model_spec.artifact.path, device)
            return CenterPointONNXInferencePipeline(
                self.pytorch_model,
                onnx_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
            )

        if backend is Backend.TENSORRT:
            logger.info("Creating CenterPoint TensorRT pipeline from %s on %s", model_spec.artifact.path, device)
            return CenterPointTensorRTInferencePipeline(
                self.pytorch_model,
                tensorrt_dir=model_spec.artifact.path,
                device=device,
                components_cfg=self._components_cfg,
            )

        raise ValueError(f"Unsupported backend: {backend.value}")
