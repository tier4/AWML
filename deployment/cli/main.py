"""
Single deployment entrypoint.

Usage:
    python -m deployment.cli.main <project> <deploy_cfg.py> <model_cfg.py> [project-specific args]
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from typing import List

import deployment.projects as projects_pkg
from deployment.core.config.base_config import parse_base_args
from deployment.projects import project_registry


def _discover_project_packages() -> List[str]:
    """Discover project package names under deployment.projects (without importing them)."""

    names: List[str] = []
    for mod in pkgutil.iter_modules(projects_pkg.__path__):
        if not mod.ispkg:
            continue
        if mod.name.startswith("_"):
            continue
        names.append(mod.name)
    return sorted(names)


def _import_and_register_project(project_name: str) -> None:
    """Import project package, which should register itself into project_registry."""
    importlib.import_module(f"deployment.projects.{project_name}")


def build_parser() -> argparse.ArgumentParser:
    """Build the unified deployment CLI parser.

    This discovers `deployment.projects.<name>` bundles, imports them to trigger
    registration into `deployment.projects.project_registry`, then creates a
    subcommand per registered project.
    """
    parser = argparse.ArgumentParser(
        description="AWML Deployment CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="project", required=True)

    # Discover projects and import them so they can contribute args.
    for project_name in _discover_project_packages():
        try:
            _import_and_register_project(project_name)
        except Exception:
            # Skip broken/incomplete project bundles rather than breaking the whole CLI.
            continue

        try:
            adapter = project_registry.get(project_name)
        except KeyError:
            continue

        sub = subparsers.add_parser(project_name, help=f"{project_name} deployment")
        parse_base_args(sub)  # adds deploy_cfg, model_cfg, --log-level
        adapter.add_args(sub)
        sub.set_defaults(_adapter_name=project_name)

    return parser


def main(argv: List[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argv list (without program name). If None, uses `sys.argv[1:]`.

    Returns:
        Process exit code (0 for success).
    """
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)

    adapter = project_registry.get(args._adapter_name)
    return int(adapter.run(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
