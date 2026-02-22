"""Deployment project bundles.

Each subpackage under `deployment/projects/<project>/` should register a
`ProjectAdapter` into `deployment.projects.registry.project_registry`.
"""

from deployment.projects.registry import ProjectAdapter, project_registry

__all__ = ["ProjectAdapter", "project_registry"]
