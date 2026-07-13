"""
Project registry for deployment bundles.

Each deployment project registers an adapter that knows how to:
- construct data_loader / evaluator / runner
- execute the deployment workflow

This keeps `deployment/cli/main.py` project-agnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class ProjectAdapter:
    """Minimal adapter interface for a deployment project.

    Projects deliberately have no per-project CLI flags: everything that shapes the exported
    artifact lives in the deploy config so it is versioned with the artifact and reproducible.
    The CLI only carries invocation concerns (``deploy_cfg``, ``model_cfg``, ``--log-level``),
    which ``deployment/cli/args.py`` adds to every subparser.

    Required-component validation is the deploy config's job (each project's
    ``*DeploymentConfig._validate_components``), so the adapter only maps a name to its ``run``.
    """

    name: str
    run: Callable[[argparse.Namespace], int]


class ProjectRegistry:
    """In-memory registry of deployment project adapters.

    The unified CLI discovers and imports `deployment.projects.<name>` packages;
    each package registers a `ProjectAdapter` here. This keeps core/cli code
    project-agnostic while enabling project-specific argument wiring and run logic.
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, ProjectAdapter] = {}

    def register(self, adapter: ProjectAdapter) -> None:
        name = adapter.name.strip().lower()
        if not name:
            raise ValueError("ProjectAdapter.name must be non-empty")
        self._adapters[name] = adapter

    def get(self, name: str) -> ProjectAdapter:
        key = (name or "").strip().lower()
        if key not in self._adapters:
            available = ", ".join(sorted(self._adapters.keys()))
            raise KeyError(f"Unknown project '{name}'. Available: [{available}]")
        return self._adapters[key]

    def list_projects(self) -> list[str]:
        return sorted(self._adapters.keys())


project_registry = ProjectRegistry()
