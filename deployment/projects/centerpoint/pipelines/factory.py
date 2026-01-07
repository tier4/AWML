"""
CenterPoint Pipeline Factory.

Registers CenterPoint pipelines into the global pipeline_registry so evaluators can create pipelines
via `deployment.pipelines.factory.PipelineFactory`.
"""

import logging
from typing import Any, Mapping, Optional

from deployment.core.backend import Backend
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.base_factory import BasePipelineFactory
from deployment.pipelines.base_pipeline import BaseDeploymentPipeline
from deployment.pipelines.registry import pipeline_registry
from deployment.projects.centerpoint.pipelines.onnx import CenterPointONNXPipeline
from deployment.projects.centerpoint.pipelines.pytorch import CenterPointPyTorchPipeline
from deployment.projects.centerpoint.pipelines.tensorrt import CenterPointTensorRTPipeline

logger = logging.getLogger(__name__)


@pipeline_registry.register
class CenterPointPipelineFactory(BasePipelineFactory):
    """Pipeline factory for CenterPoint across supported backends.

    Supports passing `components_cfg` to configure component file paths
    and IO specifications.
    """

    @classmethod
    def get_project_name(cls) -> str:
        return "centerpoint"

    @classmethod
    def create_pipeline(
        cls,
        model_spec: ModelSpec,
        pytorch_model: Any,
        device: Optional[str] = None,
        components_cfg: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> BaseDeploymentPipeline:
        """Create a CenterPoint pipeline for the specified backend.

        Args:
            model_spec: Model specification (backend/device/path)
            pytorch_model: PyTorch model instance for preprocessing
            device: Override device (uses model_spec.device if None)
            components_cfg: Unified component configuration dict from deploy_config.
                           Used to configure component file paths.
            **kwargs: Additional arguments (unused)

        Returns:
            Pipeline instance for the specified backend
        """
        device = device or model_spec.device
        backend = model_spec.backend

        cls._validate_backend(backend)

        if backend is Backend.PYTORCH:
            logger.info(f"Creating CenterPoint PyTorch pipeline on {device}")
            return CenterPointPyTorchPipeline(pytorch_model, device=device)

        if backend is Backend.ONNX:
            logger.info(f"Creating CenterPoint ONNX pipeline from {model_spec.path} on {device}")
            return CenterPointONNXPipeline(
                pytorch_model,
                onnx_dir=model_spec.path,
                device=device,
                components_cfg=components_cfg,
            )

        if backend is Backend.TENSORRT:
            logger.info(f"Creating CenterPoint TensorRT pipeline from {model_spec.path} on {device}")
            return CenterPointTensorRTPipeline(
                pytorch_model,
                tensorrt_dir=model_spec.path,
                device=device,
                components_cfg=components_cfg,
            )

        raise ValueError(f"Unsupported backend: {backend.value}")
