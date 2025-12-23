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
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class ProjectAdapter:
    """Minimal adapter interface for a deployment project."""

    name: str
    add_args: Callable  # (argparse.ArgumentParser) -> None
    run: Callable  # (argparse.Namespace) -> int


class ProjectRegistry:
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

    def list(self) -> list[str]:
        return sorted(self._adapters.keys())


project_registry = ProjectRegistry()
