from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class VoxelDict(TypedDict):
    """Voxelization output from CenterPoint feature extraction (ONNX/export path).

    Matches the dict returned alongside ``input_features`` from ``_extract_features``.

    Attributes:
        voxels: Packed voxel feature tensor.
        num_points: Per-voxel point counts.
        coors: Voxel coordinates (e.g. batch and grid indices).
    """

    voxels: torch.Tensor
    num_points: torch.Tensor
    coors: torch.Tensor


@dataclass(frozen=True)
class CenterPointFeatureSample:
    """Immutable bundle of backbone inputs and sparse tensor layout for export.

    Built by `deployment.projects.centerpoint.export.sample_extractor.CenterPointSampleExtractor`
    for ONNX/TensorRT pipelines that need validated tensors and a consistent voxel dict.

    Attributes:
        input_features: Tensor fed to the rest of the network after voxelization.
        voxel_dict: Sparse structure with keys ``voxels``, ``num_points``, ``coors``.
    """

    input_features: torch.Tensor
    voxel_dict: VoxelDict

    @property
    def coors(self) -> torch.Tensor:
        return self.voxel_dict["coors"]


def compute_batch_size(coors: torch.Tensor) -> int:
    """Infer batch size from voxel coordinates.

    Assumes the batch index is column 0 and rows are sorted by batch index (the
    layout produced by mmdet3d voxelization). Returns 1 for an empty tensor.
    """
    if len(coors) == 0:
        return 1
    return int(coors[-1, 0].item()) + 1
