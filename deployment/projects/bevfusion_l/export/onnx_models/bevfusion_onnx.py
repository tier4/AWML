"""BEVFusion deploy-only ONNX module wrappers.

These wrappers adapt BEVFusion submodules to the ONNX export interface (fixed input/output
signatures) and are the ``module`` fed to the shared ``OnnxExportPipeline`` via
``BEVFusionComponentBuilder``. They replicate the containers from the legacy
``projects/BEVFusion/deploy/containers.py`` within the new deployment framework.

- :class:`BEVFusionSparseWrapper`: LiDAR sparse encoder only (voxels/coors/num_points -> BEV feature map).
- :class:`BEVFusionDenseWrapper`: SECOND + neck + head (+ postprocess) on a BEV feature map.

The full LiDAR graph (voxels/coors/num_points -> detection triple) is not wrapped here: it is
composed from the exported sparse + dense ONNX by the merge finalize hook (see ``transforms.py``),
so no single full-graph module wrapper is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deployment.projects.bevfusion_l.io.head_outputs import head_dict_to_detection_outputs
from deployment.projects.bevfusion_l.io.voxel_inputs import graph_input_zyx_to_model_indices_xyz


def normalize_sparse_coors_for_autoware(coors: torch.Tensor) -> torch.Tensor:
    """Normalize sparse coordinates to the legacy Autoware export contract.

    Graph **inputs** must be ``[z, y, x]`` (no batch). This wrapper flips to
    ``[x, y, z]`` and prepends batch — same as ``projects/BEVFusion/deploy/containers.py``.
    Voxelization outputs ``[x, y, z]``; convert with ``voxel_indices_xyz_to_graph_input_zyx``
    before tracing or feeding ONNX/TRT.
    """
    # Guard that spconv gets int32 indices. Conditional so tracing an already-int32 input (the
    # normal case — the graph input is declared int32) emits no redundant no-op Cast in the ONNX.
    if coors.dtype != torch.int32:
        coors = coors.to(dtype=torch.int32)
    if coors.shape[1] == 3:
        num_points = coors.shape[0]
        coors = graph_input_zyx_to_model_indices_xyz(coors)
        batch_coors = torch.zeros(num_points, 1, dtype=torch.int32, device=coors.device)
        coors = torch.cat([batch_coors, coors], dim=1).contiguous()
    return coors


class BEVFusionSparseWrapper(nn.Module):
    """LiDAR sparse encoder only: voxels/coors/num_points → BEV feature map."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.mod = model

    def forward(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> torch.Tensor:
        # Keep voxels FP32 for spconv. Conditional so an already-FP32 trace input (the normal
        # case — the graph input is declared float32) emits no redundant no-op Cast in the ONNX.
        if voxels.dtype != torch.float32:
            voxels = voxels.to(dtype=torch.float32)
        coors = normalize_sparse_coors_for_autoware(coors)

        return self.mod.extract_pts_feat(voxels, coors, num_points_per_voxel, points=None)


class BEVFusionDenseWrapper(nn.Module):
    """SECOND + neck + head (+ ONNX postprocess). Input: ``lidar_bev`` [B,C,H,W]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.mod = model

    def forward(self, lidar_bev: torch.Tensor) -> tuple:
        x = lidar_bev
        if self.mod.pts_backbone is not None:
            x = self.mod.pts_backbone(x)
        if self.mod.pts_neck is not None:
            x = self.mod.pts_neck(x)
        x = self.mod._align_lidar_bev_to_head_grid(x)
        outputs = self.mod.bbox_head(x, [])
        head_out = outputs[0][0]
        return head_dict_to_detection_outputs(head_out)
