"""
CenterPoint Pipeline Factory.

This module provides the factory for creating CenterPoint pipelines
across different backends (PyTorch, ONNX, TensorRT).
"""

import logging
from typing import Any, Optional

from deployment.core.backend import Backend
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.centerpoint.centerpoint_onnx import CenterPointONNXPipeline
from deployment.pipelines.centerpoint.centerpoint_pytorch import CenterPointPyTorchPipeline
from deployment.pipelines.centerpoint.centerpoint_tensorrt import CenterPointTensorRTPipeline
from deployment.pipelines.common.base_factory import BasePipelineFactory
from deployment.pipelines.common.base_pipeline import BaseDeploymentPipeline
from deployment.pipelines.common.project_names import ProjectNames
from deployment.pipelines.common.registry import pipeline_registry

logger = logging.getLogger(__name__)


@pipeline_registry.register
class CenterPointPipelineFactory(BasePipelineFactory):
    """
    Factory for creating CenterPoint deployment pipelines.

    Supports PyTorch, ONNX, and TensorRT backends for 3D object detection.

    Example:
        >>> from deployment.pipelines.centerpoint.factory import CenterPointPipelineFactory
        >>> pipeline = CenterPointPipelineFactory.create_pipeline(
        ...     model_spec=model_spec,
        ...     pytorch_model=model,
        ... )
    """

    @classmethod
    def get_project_name(cls) -> str:
        """Return the project name for registry lookup."""
        return ProjectNames.CENTERPOINT

    @classmethod
    def create_pipeline(
        cls,
        model_spec: ModelSpec,
        pytorch_model: Any,
        device: Optional[str] = None,
        **kwargs,
    ) -> BaseDeploymentPipeline:
        """
        Create a CenterPoint pipeline for the specified backend.

        Args:
            model_spec: Model specification (backend/device/path)
            pytorch_model: PyTorch CenterPoint model instance
            device: Override device (uses model_spec.device if None)
            **kwargs: Additional arguments (unused for CenterPoint)

        Returns:
            CenterPoint pipeline instance

        Raises:
            ValueError: If backend is not supported
        """
        device = device or model_spec.device
        backend = model_spec.backend

        cls._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info(f"Creating CenterPoint PyTorch pipeline on {device}")
            return CenterPointPyTorchPipeline(pytorch_model, device=device)

        elif backend is Backend.ONNX:
            logger.info(f"Creating CenterPoint ONNX pipeline from {model_spec.path} on {device}")
            return CenterPointONNXPipeline(
                pytorch_model,
                onnx_dir=model_spec.path,
                device=device,
            )

        elif backend is Backend.TENSORRT:
            logger.info(f"Creating CenterPoint TensorRT pipeline from {model_spec.path} on {device}")
            return CenterPointTensorRTPipeline(
                pytorch_model,
                tensorrt_dir=model_spec.path,
                device=device,
            )

        else:
            raise ValueError(f"Unsupported backend: {backend.value}")
