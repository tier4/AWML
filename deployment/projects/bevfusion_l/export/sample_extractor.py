"""BEVFusion export sample extractor.

Produces the typed tracing sample consumed by :class:`BEVFusionComponentBuilder`: it loads a
point-cloud sample and runs BEVFusion voxelization, returning the voxel features, the sparse
coordinates in the ONNX graph-input ``[z, y, x]`` layout, and the per-voxel point counts.
"""

from __future__ import annotations

import torch

from deployment.export.pipelines.sample_extractor import SampleExtractor
from deployment.io.base_data_loader import BaseDataLoader
from deployment.projects.bevfusion_l.io.sample_types import BEVFusionVoxelSample
from deployment.projects.bevfusion_l.io.voxel_inputs import voxel_indices_xyz_to_graph_input_zyx


class BEVFusionSampleExtractor(SampleExtractor):
    """Extract a voxelized BEVFusion sample for ONNX export tracing."""

    def extract_sample(
        self,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        sample_idx: int,
    ) -> BEVFusionVoxelSample:
        """Load a sample and voxelize it into a typed tracing payload.

        Args:
            model: BEVFusion model exposing ``pts_voxel_layer`` (used for voxelization).
            data_loader: Loader providing ``load_sample(sample_idx)`` with a ``points`` tensor.
            sample_idx: Index of the sample to trace with.

        Returns:
            A :class:`BEVFusionVoxelSample` with tensors on the model's device.
        """
        sample = data_loader.load_sample(sample_idx)
        points = sample["points"]

        device = next(model.parameters()).device
        points = points.to(device).float()

        with torch.no_grad():
            ret = model.pts_voxel_layer(points)
            if len(ret) == 3:
                feats, coords, sizes = ret
            else:
                feats, coords = ret
                sizes = torch.ones(feats.shape[0], device=device)

            coords = coords[:, :].to(dtype=torch.int32)  # [M, 3] (x, y, z) from voxel layer
            coords = voxel_indices_xyz_to_graph_input_zyx(coords)  # ONNX graph input: [z, y, x]

        return BEVFusionVoxelSample(voxels=feats, coors=coords, num_points_per_voxel=sizes)
