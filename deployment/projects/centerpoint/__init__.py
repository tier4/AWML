"""CenterPoint deployment bundle.

This package owns all CenterPoint deployment-specific code (runner/evaluator/loader/pipelines/export).
It registers a ProjectAdapter into the global `project_registry` so the unified CLI can invoke it.
"""

from __future__ import annotations

from deployment.projects.centerpoint.cli import add_args
from deployment.projects.centerpoint.entrypoint import run

# Trigger pipeline factory registration for this project.
from deployment.projects.centerpoint.pipelines.factory import CenterPointPipelineFactory  # noqa: F401
from deployment.projects.registry import ProjectAdapter, project_registry

project_registry.register(
    ProjectAdapter(
        name="centerpoint",
        add_args=add_args,
        run=run,
    )
)
