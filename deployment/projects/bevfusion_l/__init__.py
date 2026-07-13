"""BEVFusion deployment bundle.

Registers a ProjectAdapter into the global project_registry so the unified CLI can invoke it.
Supports LiDAR BEVFusion export to ONNX/TensorRT with evaluation and verification.
"""

from __future__ import annotations

from deployment.projects.bevfusion_l.entrypoint import run
from deployment.projects.registry import ProjectAdapter, project_registry

project_registry.register(
    ProjectAdapter(
        name="bevfusion_l",
        run=run,
    )
)
