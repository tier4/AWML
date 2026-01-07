"""
Artifact path helpers for CenterPoint pipelines.

Resolves component artifact paths from the unified deploy config.
"""

from __future__ import annotations

import os.path as osp
from typing import Any, Mapping


def resolve_component_artifact_path(
    *,
    base_dir: str,
    components_cfg: Mapping[str, Any],
    component: str,
    file_key: str,
    default_filename: str,
) -> str:
    """Resolve an artifact path for one component.

    Args:
        base_dir: Base directory for artifacts (onnx_dir or tensorrt_dir)
        components_cfg: The unified `components` dict from deploy_config
        component: Component name (e.g., 'voxel_encoder', 'backbone_head')
        file_key: Key to look up ('onnx_file' or 'engine_file')
        default_filename: Fallback filename if not specified in config

    Returns:
        Absolute path to the artifact file

    Priority:
    - `components_cfg[component][file_key]` if present
    - otherwise `default_filename`

    Supports absolute paths in config; otherwise joins with `base_dir`.
    """
    comp_cfg = components_cfg.get(component, {}) or {}
    if not isinstance(comp_cfg, Mapping):
        raise TypeError(f"components['{component}'] must be a mapping, got {type(comp_cfg)}")

    filename = comp_cfg.get(file_key, default_filename)
    if not isinstance(filename, str) or not filename:
        raise TypeError(f"components['{component}']['{file_key}'] must be a non-empty str, got {type(filename)}")

    return filename if osp.isabs(filename) else osp.join(base_dir, filename)


def get_component_files(components_cfg: Mapping[str, Any], file_key: str) -> dict[str, str]:
    """Get all component filenames for a given file type.

    Args:
        components_cfg: The unified `components` dict from deploy_config
        file_key: Key to look up ('onnx_file' or 'engine_file')

    Returns:
        Dict mapping component name to filename
    """
    result = {}
    for comp_name, comp_cfg in components_cfg.items():
        if isinstance(comp_cfg, Mapping) and file_key in comp_cfg:
            result[comp_name] = comp_cfg[file_key]
    return result
