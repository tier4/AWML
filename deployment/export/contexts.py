"""
Typed context objects for deployment workflows.

Usage:
    # Create context for export
    ctx = ExportContext()

    # Pass to orchestrator
    result = export_orchestrator.run(ctx)

Project-specific subclasses live with their project (e.g.
``deployment.projects.centerpoint.contexts.CenterPointExportContext``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExportContext:
    """
    Base context for export operations.

    Marker base class for export contexts; project-specific subclasses add
    typed fields for their export parameters.
    """
