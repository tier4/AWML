"""
Artifact Path Resolution for Deployment Pipelines.

This module provides:
1. Artifact dataclass - represents an exported model artifact
2. Path resolution functions - resolve artifact paths from deploy config

Supports:
- Single-component models (YOLOX, Calibration): use component="model"
- Multi-component models (CenterPoint): use component="voxel_encoder", "backbone_head", etc.
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
        multi_file: True if artifact is a directory containing multiple files
                    (e.g., CenterPoint has voxel_encoder.onnx + backbone_head.onnx).
    """

    path: str
    multi_file: bool = False

    def exists(self) -> bool:
        """Check if the artifact exists on disk."""
        return os.path.exists(self.path)

    def is_directory(self) -> bool:
        """Check if the artifact is a directory."""
        return os.path.isdir(self.path)

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
    component: str,
    file_key: str,
) -> str:
    """Resolve artifact path for any component.

    This is the entry point for artifact path resolution.

    Args:
        base_dir: Base directory for artifacts (onnx_dir or tensorrt_dir),
                  or direct path to an artifact file.
        components_cfg: The `components` dict from deploy_config.
                       Can be None for backwards compatibility.
        component: Component name (e.g., 'model', 'voxel_encoder', 'backbone_head')
        file_key: Key to look up ('onnx_file' or 'engine_file')
        default_filename: Fallback filename if not specified in config

    Returns:
        Resolved path to the artifact file

    Resolution strategy (single supported mode):
    1. `base_dir` must be a directory (e.g., `.../onnx` or `.../tensorrt`)
    2. Require `components_cfg[component][file_key]` to be set
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
            component="model",
            file_key="onnx_file",
        )

        # Multi-component model (CenterPoint)
        resolve_artifact_path(
            base_dir="work_dirs/centerpoint/tensorrt",
            components_cfg={"voxel_encoder": {"engine_file": "voxel.engine"}},
            component="voxel_encoder",
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
    filename = _get_filename_from_config(components_cfg, component, file_key)
    if not filename:
        raise KeyError(
            "Missing artifact filename in deploy config. "
            f"Expected components['{component}']['{file_key}'] to be set."
        )

    if osp.isabs(filename):
        raise ValueError(
            "Absolute artifact paths are not allowed. "
            f"Set components['{component}']['{file_key}'] to a relative filename under base_dir instead. "
            f"(got: {filename})"
        )

    base_abs = osp.abspath(base_dir)
    path = osp.abspath(osp.join(base_abs, filename))
    # Prevent escaping base_dir via '../'
    if osp.commonpath([base_abs, path]) != base_abs:
        raise ValueError(
            "Artifact path must stay within base_dir. "
            f"Got components['{component}']['{file_key}']={filename} which resolves to {path} outside {base_abs}."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Configured artifact file not found: {path}. "
            f"(base_dir={base_dir}, component={component}, file_key={file_key})"
        )
    return path


def _get_filename_from_config(
    components_cfg: Optional[Mapping[str, Any]],
    component: str,
    file_key: str,
) -> Optional[str]:
    """Extract filename from components config."""
    if not components_cfg:
        return None

    comp_cfg = components_cfg.get(component, {})
    if not isinstance(comp_cfg, Mapping):
        return None

    filename = comp_cfg.get(file_key)
    if isinstance(filename, str) and filename:
        return filename
    return None


def _search_directory_for_artifact(
    directory: str,
    file_key: str,
    default_filename: str,
) -> Optional[str]:
    """Search directory for matching artifact file."""
    ext = FILE_EXTENSIONS.get(file_key, "")
    if not ext:
        return None

    try:
        matching_files = [f for f in os.listdir(directory) if f.endswith(ext)]
    except OSError:
        return None

    if not matching_files:
        return None

    # Single file: use it
    if len(matching_files) == 1:
        resolved = osp.join(directory, matching_files[0])
        logger.info(f"Resolved artifact path: {directory} -> {resolved}")
        return resolved

    # Multiple files: prefer default_filename
    if default_filename in matching_files:
        resolved = osp.join(directory, default_filename)
        logger.info(f"Resolved artifact path using default: {resolved}")
        return resolved

    # Otherwise use the first one (with warning)
    resolved = osp.join(directory, matching_files[0])
    logger.warning(f"Multiple {ext} files found in {directory}, using first one: {matching_files[0]}")
    return resolved


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
        >>> components = {"voxel_encoder": {"onnx_file": "voxel.onnx"},
        ...               "backbone_head": {"onnx_file": "head.onnx"}}
        >>> get_component_files(components, "onnx_file")
        {"voxel_encoder": "voxel.onnx", "backbone_head": "head.onnx"}
    """
    result = {}
    for comp_name, comp_cfg in components_cfg.items():
        if isinstance(comp_cfg, Mapping) and file_key in comp_cfg:
            result[comp_name] = comp_cfg[file_key]
    return result


# Convenience aliases for common use cases
def resolve_onnx_path(
    base_dir: str,
    components_cfg: Optional[Mapping[str, Any]] = None,
    component: str = "model",
) -> str:
    """Convenience function for resolving ONNX paths."""
    return resolve_artifact_path(
        base_dir=base_dir,
        components_cfg=components_cfg,
        component=component,
        file_key="onnx_file",
    )


def resolve_engine_path(
    base_dir: str,
    components_cfg: Optional[Mapping[str, Any]] = None,
    component: str = "model",
) -> str:
    """Convenience function for resolving TensorRT engine paths."""
    return resolve_artifact_path(
        base_dir=base_dir,
        components_cfg=components_cfg,
        component=component,
        file_key="engine_file",
    )
