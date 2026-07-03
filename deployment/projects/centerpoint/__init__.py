"""CenterPoint deployment bundle.

Import concrete modules (``deployment.projects.centerpoint.runner``, …). This ``__init__`` only
registers the project with ``deployment.projects.registry`` when the package is imported.
"""

from __future__ import annotations

from deployment.projects.centerpoint.cli import add_args
from deployment.projects.centerpoint.entrypoint import run
from deployment.projects.registry import ProjectAdapter, project_registry

project_registry.register(
    ProjectAdapter(
        name="centerpoint",
        add_args=add_args,
        run=run,
        required_components=("pts_voxel_encoder", "pts_backbone_neck_head"),
    )
)
