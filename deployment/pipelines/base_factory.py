"""
Base Pipeline Factory for Project-specific Pipeline Creation.

Flattened from `deployment/pipelines/common/base_factory.py`.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from deployment.core.backend import Backend
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class BasePipelineFactory(ABC):
    @classmethod
    @abstractmethod
    def get_project_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_pipeline(
        cls,
        model_spec: ModelSpec,
        pytorch_model: Any,
        device: Optional[str] = None,
        **kwargs,
    ) -> BaseDeploymentPipeline:
        raise NotImplementedError

    @classmethod
    def get_supported_backends(cls) -> list:
        return [Backend.PYTORCH, Backend.ONNX, Backend.TENSORRT]

    @classmethod
    def _validate_backend(cls, backend: Backend) -> None:
        supported = cls.get_supported_backends()
        if backend not in supported:
            supported_names = [b.value for b in supported]
            raise ValueError(
                f"Unsupported backend '{backend.value}' for {cls.get_project_name()}. Supported backends: {supported_names}"
            )
