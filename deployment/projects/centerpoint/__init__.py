"""CenterPoint deployment bundle.

Import concrete modules (``deployment.projects.centerpoint.runner``, …). This ``__init__`` only
registers the project with ``deployment.projects.registry`` when the package is imported.
"""

from __future__ import annotations

from deployment.projects.centerpoint.entrypoint import run
from deployment.projects.registry import ProjectAdapter, project_registry

# Options that shape the exported graph (e.g. ``rot_y_axis_reference``) live in the deploy config
# (see ``CenterPointDeploymentConfig``) so they are versioned with the artifact, not passed on the CLI.
project_registry.register(
    ProjectAdapter(
        name="centerpoint",
        run=run,
    )
)
