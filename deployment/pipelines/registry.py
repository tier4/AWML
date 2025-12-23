"""
Pipeline Registry for Dynamic Project Pipeline Registration.

Flattened from `deployment/pipelines/common/registry.py`.
"""

import logging
from typing import Any, Dict, Optional, Type

from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.pipelines.base_factory import BasePipelineFactory
from deployment.pipelines.base_pipeline import BaseDeploymentPipeline

logger = logging.getLogger(__name__)


class PipelineRegistry:
    def __init__(self):
        self._factories: Dict[str, Type[BasePipelineFactory]] = {}

    def register(self, factory_cls: Type[BasePipelineFactory]) -> Type[BasePipelineFactory]:
        if not issubclass(factory_cls, BasePipelineFactory):
            raise TypeError(f"Factory class must inherit from BasePipelineFactory, got {factory_cls.__name__}")

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
        if project_name not in self._factories:
            available = list(self._factories.keys())
            raise KeyError(f"No factory registered for project '{project_name}'. Available projects: {available}")
        return self._factories[project_name]

    def create_pipeline(
        self,
        project_name: str,
        model_spec: ModelSpec,
        pytorch_model: Any,
        device: Optional[str] = None,
        **kwargs,
    ) -> BaseDeploymentPipeline:
        factory = self.get_factory(project_name)
        return factory.create_pipeline(
            model_spec=model_spec,
            pytorch_model=pytorch_model,
            device=device,
            **kwargs,
        )

    def list_projects(self) -> list:
        return list(self._factories.keys())

    def is_registered(self, project_name: str) -> bool:
        return project_name in self._factories

    def reset(self) -> None:
        self._factories.clear()


pipeline_registry = PipelineRegistry()
