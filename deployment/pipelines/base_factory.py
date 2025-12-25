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
    """Project-specific factory interface for building deployment pipelines.

    A project registers a subclass into `deployment.pipelines.registry.pipeline_registry`.
    Evaluators then call into the registry/factory to instantiate the correct pipeline
    for a given (project, backend) pair.
    """

    @classmethod
    @abstractmethod
    def get_project_name(cls) -> str:
        """Return the unique project identifier used for registry lookup."""
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
        """Build and return a pipeline instance for the given model spec.

        Implementations typically:
        - Validate/dispatch based on `model_spec.backend`
        - Wrap `pytorch_model` or load an ONNX/TensorRT runtime
        - Construct a `BaseDeploymentPipeline` subclass configured for the backend

        Args:
            model_spec: Describes the model path/device/backend and any metadata.
            pytorch_model: A loaded PyTorch model (used for PYTORCH backends).
            device: Optional device override (defaults to `model_spec.device`).
            **kwargs: Project-specific options passed from evaluator/CLI.
        """
        raise NotImplementedError

    @classmethod
    def get_supported_backends(cls) -> list:
        """Return the list of backends this project factory can instantiate."""
        return [Backend.PYTORCH, Backend.ONNX, Backend.TENSORRT]

    @classmethod
    def _validate_backend(cls, backend: Backend) -> None:
        """Raise a ValueError if `backend` is not supported by this factory."""
        supported = cls.get_supported_backends()
        if backend not in supported:
            supported_names = [b.value for b in supported]
            raise ValueError(
                f"Unsupported backend '{backend.value}' for {cls.get_project_name()}. Supported backends: {supported_names}"
            )
