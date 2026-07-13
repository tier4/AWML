"""BEVFusion typed export/tracing sample.

Mirrors CenterPoint's ``io/sample_types.py``: the typed payload produced by the sample extractor
and consumed by the component builder lives in ``io`` (not in ``export``), so both projects keep
their typed samples in the same place.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BEVFusionVoxelSample:
    """Voxelized, export-ready BEVFusion tracing sample.

    Produced by :class:`~deployment.projects.bevfusion_l.export.sample_extractor.BEVFusionSampleExtractor`.
    All tensors are already on the model's device and in the exported graph's expected dtype/layout,
    so consumers (the component builder) use them directly and never re-handle device or dtype.

    Attributes:
        voxels: Voxel features ``[M, ...]`` (float32), on the model device.
        coors: Sparse coordinates ``[M, 3]`` in graph-input ``[z, y, x]`` layout (int32), on the model device.
        num_points_per_voxel: Per-voxel point counts ``[M]``, on the model device.
    """

    voxels: torch.Tensor
    coors: torch.Tensor
    num_points_per_voxel: torch.Tensor
