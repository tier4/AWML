"""
Pipeline Registry for Dynamic Project Pipeline Registration.

This module provides a registry pattern for managing pipeline factories,
allowing projects to register themselves and be discovered at runtime.

Usage:
    # In project's factory module (e.g., centerpoint/factory.py):
    from deployment.pipelines.common.registry import pipeline_registry

    @pipeline_registry.register
    class CenterPointPipelineFactory(BasePipelineFactory):
        ...

    # In main code:
    pipeline = pipeline_registry.create_pipeline(ProjectNames.CENTERPOINT, model_spec, pytorch_model)
"""

import logging
from typing import Any, Dict, Optional, Type

from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.common.base_factory import BasePipelineFactory
from deployment.pipelines.common.base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """
    Registry for project-specific pipeline factories.

    This registry maintains a mapping of project names to their factory classes,
    enabling dynamic pipeline creation without hardcoding project-specific logic.

    Example:
        # Register a factory
        @pipeline_registry.register
        class MyProjectPipelineFactory(BasePipelineFactory):
            @classmethod
            def get_project_name(cls) -> str:
                return "my_project"
            ...

        # Create a pipeline
        pipeline = pipeline_registry.create_pipeline(
            "my_project", model_spec, pytorch_model
        )
    """

    def __init__(self):
        self._factories: Dict[str, Type[BasePipelineFactory]] = {}

    def register(self, factory_cls: Type[BasePipelineFactory]) -> Type[BasePipelineFactory]:
        """
        Register a pipeline factory class.

        Can be used as a decorator or called directly.

        Args:
            factory_cls: Factory class implementing BasePipelineFactory

        Returns:
            The registered factory class (for decorator usage)

        Example:
            @pipeline_registry.register
            class CenterPointPipelineFactory(BasePipelineFactory):
                ...
        """
        if not issubclass(factory_cls, BasePipelineFactory):
            raise TypeError(f"Factory class must inherit from BasePipelineFactory, " f"got {factory_cls.__name__}")

        project_name = factory_cls.get_project_name()

        if project_name in self._factories:
            logger.warning(
                f"Overwriting existing factory for project '{project_name}': "
                f"{self._factories[project_name].__name__} -> {factory_cls.__name__}"
            )

        self._factories[project_name] = factory_cls
        logger.debug(f"Registered pipeline factory: {project_name} -> {factory_cls.__name__}")

        return factory_cls

    def get_factory(self, project_name: str) -> Type[BasePipelineFactory]:
        """
        Get the factory class for a project.

        Args:
            project_name: Name of the project

        Returns:
            Factory class for the project

        Raises:
            KeyError: If project is not registered
        """
        if project_name not in self._factories:
            available = list(self._factories.keys())
            raise KeyError(f"No factory registered for project '{project_name}'. " f"Available projects: {available}")

        return self._factories[project_name]

    def create_pipeline(
        self,
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
        """
        factory = self.get_factory(project_name)
        return factory.create_pipeline(
            model_spec=model_spec,
            pytorch_model=pytorch_model,
            device=device,
            **kwargs,
        )

    def list_projects(self) -> list:
        """
        List all registered projects.

        Returns:
            List of registered project names
        """
        return list(self._factories.keys())

    def is_registered(self, project_name: str) -> bool:
        """
        Check if a project is registered.

        Args:
            project_name: Name of the project

        Returns:
            True if project is registered
        """
        return project_name in self._factories

    def reset(self) -> None:
        """
        Reset the registry (mainly for testing).
        """
        self._factories.clear()


# Global registry instance
pipeline_registry = PipelineRegistry()
