"""
Artifact Path Resolution for Deployment Pipelines.

This module provides:
1. Artifact dataclass - represents an exported model artifact
2. Path resolution functions - resolve artifact paths from deploy config

Supports:
- Single-component models (YOLOX, Calibration): use component_name="model"
- Multi-component models (CenterPoint): use component_name="pts_voxel_encoder", "pts_backbone_neck_head", etc.
"""

from __future__ import annotations

import logging
import os
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Artifact Dataclass
# ============================================================================


@dataclass(frozen=True)
class Artifact:
    """
    Represents an exported model artifact (ONNX file, TensorRT engine, etc.).

    Attributes:
        path: Filesystem path to the artifact (file or directory).
    """

    path: str

    @property
    def exists(self) -> bool:
        """Whether the artifact exists on disk."""
        return os.path.exists(self.path)

    def __str__(self) -> str:
        return self.path


# ============================================================================
# Path Resolution Functions
# ============================================================================

# File extension mapping
FILE_EXTENSIONS: Dict[str, str] = {
    "onnx_file": ".onnx",
    "engine_file": ".engine",
}


def resolve_artifact_path(
    *,
    base_dir: str,
    components_cfg: Optional[Mapping[str, Any]],
    component_name: str,
    file_key: str,
) -> str:
    """Resolve artifact path for any component.

    This is the entry point for artifact path resolution.

    Args:
        base_dir: Base directory for artifacts (onnx_dir or tensorrt_dir),
                  or direct path to an artifact file.
        components_cfg: The `components` dict from deploy_config.
                       Can be None for backwards compatibility.
        component_name: Component id (e.g., 'model', 'pts_voxel_encoder', 'pts_backbone_neck_head')
        file_key: Key to look up ('onnx_file' or 'engine_file')

    Returns:
        Resolved path to the artifact file

    Resolution strategy (single supported mode):
    1. `base_dir` must be a directory (e.g., `.../onnx` or `.../tensorrt`)
    2. Require `components_cfg[component_name][file_key]` to be set
       - must be a relative path resolved under `base_dir`
    3. The resolved path must exist and be a file

    This function intentionally does NOT:
    - scan directories for matching extensions
    - fall back to default filenames
    - accept `base_dir` as a file path
    - accept absolute paths in `components` (enforces fully config-driven, workspace-relative artifacts)

    Examples:
        # Single-component model (YOLOX)
        resolve_artifact_path(
            base_dir="work_dirs/yolox/onnx",
            components_cfg={"model": {"onnx_file": "yolox.onnx"}},
            component_name="model",
            file_key="onnx_file",
        )

        # Multi-component model (CenterPoint)
        resolve_artifact_path(
            base_dir="work_dirs/centerpoint/tensorrt",
            components_cfg={"pts_voxel_encoder": {"engine_file": "pts_voxel_encoder.engine"}},
            component_name="pts_voxel_encoder",
            file_key="engine_file",
        )
    """
    if not os.path.isdir(base_dir):
        raise ValueError(
            "Artifact resolution requires `base_dir` to be a directory. "
            f"Got: {base_dir}. "
            "Set evaluation.backends.<backend>.{model_dir|engine_dir} to the artifact directory, "
            "and set the artifact filename in deploy config under components.*.{onnx_file|engine_file}."
        )

    # Require filename from components config
    filename = _get_filename_from_config(components_cfg, component_name, file_key)
    if not filename:
        raise KeyError(
            "Missing artifact filename in deploy config. "
            f"Expected components['{component_name}']['{file_key}'] to be set."
        )

    if osp.isabs(filename):
        raise ValueError(
            "Absolute artifact paths are not allowed. "
            f"Set components['{component_name}']['{file_key}'] to a relative filename under base_dir instead. "
            f"(got: {filename})"
        )

    base_abs = osp.abspath(base_dir)
    path = osp.abspath(osp.join(base_abs, filename))
    # Prevent escaping base_dir via '../'
    if osp.commonpath([base_abs, path]) != base_abs:
        raise ValueError(
            "Artifact path must stay within base_dir. "
            f"Got components['{component_name}']['{file_key}']={filename} which resolves to {path} outside {base_abs}."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Configured artifact file not found: {path}. "
            f"(base_dir={base_dir}, component_name={component_name}, file_key={file_key})"
        )
    return path


def _get_filename_from_config(
    components_cfg: Optional[Mapping[str, Any]],
    component_name: str,
    file_key: str,
) -> Optional[str]:
    """Extract filename from components config (dict or ComponentsConfig dataclass)."""
    if components_cfg is None:
        return None
    if hasattr(components_cfg, "get_artifact_filename"):
        out = components_cfg.get_artifact_filename(component_name, file_key)
        return out if isinstance(out, str) and out else None
    component_cfg = components_cfg.get(component_name, {})
    if not isinstance(component_cfg, Mapping):
        return None
    filename = component_cfg.get(file_key)
    if isinstance(filename, str) and filename:
        return filename
    return None


def get_component_files(
    components_cfg: Mapping[str, Any],
    file_key: str,
) -> Dict[str, str]:
    """Get all component filenames for a given file type.

    Useful for multi-component models to enumerate all artifacts.

    Args:
        components_cfg: The unified `components` dict from deploy_config
        file_key: Key to look up ('onnx_file' or 'engine_file')

    Returns:
        Dict mapping component name to filename

    Example:
        >>> components = {"pts_voxel_encoder": {"onnx_file": "pts_voxel_encoder.onnx"},
        ...               "pts_backbone_neck_head": {"onnx_file": "pts_backbone_neck_head.onnx"}}
        >>> get_component_files(components, "onnx_file")
        {"pts_voxel_encoder": "pts_voxel_encoder.onnx", "pts_backbone_neck_head": "pts_backbone_neck_head.onnx"}
    """
    result = {}
    for component_name, component_cfg in components_cfg.items():
        if isinstance(component_cfg, Mapping) and file_key in component_cfg:
            result[component_name] = component_cfg[file_key]
    return result


def resolve_onnx_path(
    base_dir: str,
    components_cfg: Optional[Mapping[str, Any]] = None,
    component_name: str = "model",
) -> str:
    """Convenience function for resolving ONNX paths."""
    return resolve_artifact_path(
        base_dir=base_dir,
        components_cfg=components_cfg,
        component_name=component_name,
        file_key="onnx_file",
    )


def resolve_engine_path(
    base_dir: str,
    components_cfg: Optional[Mapping[str, Any]] = None,
    component_name: str = "model",
) -> str:
    """Convenience function for resolving TensorRT engine paths."""
    return resolve_artifact_path(
        base_dir=base_dir,
        components_cfg=components_cfg,
        component_name=component_name,
        file_key="engine_file",
    )
