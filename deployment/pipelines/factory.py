"""
Pipeline Factory for Centralized Pipeline Instantiation.

This module provides a unified interface for creating deployment pipelines
using the registry pattern. Each project registers its own factory, and
this module provides convenience methods for pipeline creation.

Architecture:
    - Each project implements `BasePipelineFactory` in its own directory
    - Factories are registered with `pipeline_registry` using decorators
    - This factory provides a unified interface for pipeline creation

Usage:
    from deployment.pipelines.factory import PipelineFactory
    pipeline = PipelineFactory.create("centerpoint", model_spec, pytorch_model)

    # Or use registry directly:
    from deployment.pipelines.registry import pipeline_registry
    pipeline = pipeline_registry.create_pipeline("centerpoint", model_spec, pytorch_model)
"""

import logging
from typing import Any, List, Optional

from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.base_pipeline import BaseDeploymentPipeline
from deployment.pipelines.registry import pipeline_registry

logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory for creating deployment pipelines.

    This class provides a unified interface for creating pipelines across
    different projects and backends. It delegates to project-specific
    factories through the pipeline registry.

    Example:
        # Create a pipeline using the generic method
        pipeline = PipelineFactory.create("centerpoint", model_spec, pytorch_model)

        # List available projects
        projects = PipelineFactory.list_projects()
    """

    @staticmethod
    def create(
        project_name: str,
        model_spec: ModelSpec,
        pytorch_model: Any,
        device: Optional[str] = None,
        **kwargs,
    ) -> BaseDeploymentPipeline:
        """
        Create a pipeline for the specified project.

        Args:
            project_name: Name of the project (e.g., "centerpoint", "yolox")
            model_spec: Model specification (backend/device/path)
            pytorch_model: PyTorch model instance
            device: Override device (uses model_spec.device if None)
            **kwargs: Project-specific arguments

        Returns:
            Pipeline instance

        Raises:
            KeyError: If project is not registered
            ValueError: If backend is not supported

        Example:
            >>> pipeline = PipelineFactory.create(
            ...     "centerpoint",
            ...     model_spec,
            ...     pytorch_model,
            ... )
        """
        return pipeline_registry.create_pipeline(
            project_name=project_name,
            model_spec=model_spec,
            pytorch_model=pytorch_model,
            device=device,
            **kwargs,
        )

    @staticmethod
    def list_projects() -> List[str]:
        """
        List all registered projects.

        Returns:
            List of registered project names
        """
        return pipeline_registry.list_projects()

    @staticmethod
    def is_project_registered(project_name: str) -> bool:
        """
        Check if a project is registered.

        Args:
            project_name: Name of the project

        Returns:
            True if project is registered
        """
        return pipeline_registry.is_registered(project_name)
