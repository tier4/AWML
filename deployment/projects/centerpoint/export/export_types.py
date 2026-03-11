from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class VoxelDict(TypedDict):
    voxels: torch.Tensor
    num_points: torch.Tensor
    coors: torch.Tensor


@dataclass(frozen=True)
class CenterPointExportSample:
    input_features: torch.Tensor
    voxel_dict: VoxelDict

    @property
    def coors(self) -> torch.Tensor:
        return self.voxel_dict["coors"]
