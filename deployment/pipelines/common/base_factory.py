"""
Base Pipeline Factory for Project-specific Pipeline Creation.

This module provides the abstract base class for pipeline factories,
defining a unified interface for creating pipelines across different backends.

Architecture:
    - Each project (CenterPoint, YOLOX, etc.) implements its own factory
    - Factories are registered with the PipelineRegistry
    - Main factory uses registry to lookup and delegate to project factories

Benefits:
    - Open-Closed Principle: Add new projects without modifying main factory
    - Single Responsibility: Each project manages its own pipeline creation
    - Decoupled: Project-specific logic stays in project directories
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from deployment.core.backend import Backend
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.common.base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class BasePipelineFactory(ABC):
    """
    Abstract base class for project-specific pipeline factories.

    Each project (CenterPoint, YOLOX, Calibration, etc.) should implement
    this interface to provide its own pipeline creation logic.

    Example:
        class CenterPointPipelineFactory(BasePipelineFactory):
            @classmethod
            def get_project_name(cls) -> str:
                return "centerpoint"

            @classmethod
            def create_pipeline(
                cls,
                model_spec: ModelSpec,
                pytorch_model: Any,
                device: Optional[str] = None,
                **kwargs
            ) -> BaseDeploymentPipeline:
                # Create and return appropriate pipeline based on backend
                ...
    """

    @classmethod
    @abstractmethod
    def get_project_name(cls) -> str:
        """
        Get the project name for registry lookup.

        Returns:
            Project name (e.g., "centerpoint", "yolox", "calibration")
        """
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
        """
        Create a pipeline for the specified backend.

        Args:
            model_spec: Model specification (backend/device/path)
            pytorch_model: PyTorch model instance
            device: Override device (uses model_spec.device if None)
            **kwargs: Project-specific arguments

        Returns:
            Pipeline instance for the specified backend

        Raises:
            ValueError: If backend is not supported
        """
        raise NotImplementedError

    @classmethod
    def get_supported_backends(cls) -> list:
        """
        Get list of supported backends for this project.

        Override this method to specify which backends are supported.
        Default implementation returns all common backends.

        Returns:
            List of supported Backend enums
        """
        return [Backend.PYTORCH, Backend.ONNX, Backend.TENSORRT]

    @classmethod
    def _validate_backend(cls, backend: Backend) -> None:
        """
        Validate that the backend is supported.

        Args:
            backend: Backend to validate

        Raises:
            ValueError: If backend is not supported
        """
        supported = cls.get_supported_backends()
        if backend not in supported:
            supported_names = [b.value for b in supported]
            raise ValueError(
                f"Unsupported backend '{backend.value}' for {cls.get_project_name()}. "
                f"Supported backends: {supported_names}"
            )
