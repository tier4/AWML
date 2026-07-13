"""BEVFusion voxel input contract for deploy / ONNX / TensorRT.

Everything about how voxel data enters the exported graph lives here:

- The **input names** (``voxels`` / ``coors`` / ``num_points_per_voxel``) declared by the deploy
  config and baked into the ONNX / TensorRT graphs, and :func:`map_voxel_inputs` to bind the three
  arrays to a session/engine's declared inputs.
- The **coordinate layout** of ``coors``: voxelization (``pts_voxel_layer``) returns indices as
  ``[x, y, z]``; the legacy Autoware-compatible ONNX expects graph inputs as ``[z, y, x]`` (no
  batch). Inside the exported wrapper the indices are flipped back to ``[x, y, z]`` and a batch
  column is prepended before ``pts_middle_encoder`` (``sparse_shape`` is ``[H, W, D]``). PyTorch
  evaluation uses ``[batch, x, y, z]`` directly and does not use these flips.
"""

from __future__ import annotations

from typing import Dict, Sequence, TypeVar

import torch

# Canonical sparse / merged-graph voxel input names. These are declared by the deploy config
# (``components.*.io.inputs``) and baked into the exported ONNX / TensorRT graphs, so they are
# the authoritative names to feed by — no substring guessing.
VOXELS_INPUT = "voxels"
COORS_INPUT = "coors"
NUM_POINTS_INPUT = "num_points_per_voxel"

_T = TypeVar("_T")


def map_voxel_inputs(input_names: Sequence[str], *, voxels: _T, coors: _T, num_points: _T) -> Dict[str, _T]:
    """Bind the voxel/coors/num-points arrays to a model's declared input names.

    Args:
        input_names: Input tensor names reported by the ONNX session / TensorRT engine.
        voxels, coors, num_points: The three input arrays to feed.

    Returns:
        A ``{input_name: array}`` feed dict, one entry per name in ``input_names``.

    Raises:
        RuntimeError: If an input name is not one of the canonical voxel input names — surfaces
            an export/config name mismatch loudly instead of silently dropping an input.
    """
    by_name: Dict[str, _T] = {VOXELS_INPUT: voxels, COORS_INPUT: coors, NUM_POINTS_INPUT: num_points}
    feed: Dict[str, _T] = {}
    for name in input_names:
        if name not in by_name:
            raise RuntimeError(f"Unexpected model input {name!r}; expected one of {list(by_name)}")
        feed[name] = by_name[name]
    return feed


def _flip_last_axis(coors: torch.Tensor) -> torch.Tensor:
    """Reverse the axis order of ``[M, 3]`` indices, validating the shape.

    The two directional helpers below are the *same* reversal (a flip is its own inverse); the
    distinct names document the intended direction at each call site.
    """
    if coors.ndim != 2 or coors.shape[1] != 3:
        raise ValueError(f"Expected [M, 3] coors, got shape {tuple(coors.shape)}")
    return coors.flip(dims=[-1]).contiguous()


def voxel_indices_xyz_to_graph_input_zyx(coors: torch.Tensor) -> torch.Tensor:
    """``[M, 3]`` voxel indices ``[x, y, z]`` → graph input ``[z, y, x]``."""
    return _flip_last_axis(coors)


def graph_input_zyx_to_model_indices_xyz(coors: torch.Tensor) -> torch.Tensor:
    """``[M, 3]`` graph input ``[z, y, x]`` → model indices ``[x, y, z]`` (wrapper flip)."""
    return _flip_last_axis(coors)
