"""
Project registry for deployment bundles.

Each deployment project registers an adapter that knows how to:
- add its CLI args
- construct data_loader / evaluator / runner
- execute the deployment workflow

This keeps `deployment/cli/main.py` project-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple


@dataclass(frozen=True)
class ProjectAdapter:
    """Minimal adapter interface for a deployment project."""

    name: str
    add_args: Callable  # (argparse.ArgumentParser) -> None
    run: Callable  # (argparse.Namespace) -> int
    required_components: Tuple[str, ...] = ()


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

    def validate_required_components(self, project_name: str, components_cfg) -> None:
        """Validate required component keys for a registered project."""
        adapter = self.get(project_name)
        if not adapter.required_components:
            return

        missing = []
        for component_name in adapter.required_components:
            try:
                components_cfg.get_component(component_name)
            except KeyError:
                missing.append(component_name)

        if not missing:
            return

        available = sorted(list(components_cfg.component_names()))
        missing_str = ", ".join(missing)
        available_str = ", ".join(available)
        raise KeyError(
            f"{adapter.name} requires components [{missing_str}], " f"but available components are [{available_str}]."
        )


project_registry = ProjectRegistry()
