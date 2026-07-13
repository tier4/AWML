"""BEVFusion PyTorch Pipeline Implementation (sparse + dense seams)."""

from __future__ import annotations

import logging
from typing import List

import torch
from typing_extensions import override

from deployment.config.enums import Backend
from deployment.primitives.device import DeviceSpec
from deployment.projects.bevfusion_l.inference.bevfusion_inference_pipeline import BEVFusionInferencePipeline
from deployment.projects.bevfusion_l.io.head_outputs import head_dict_to_detection_outputs

logger = logging.getLogger(__name__)


class BEVFusionPyTorchInferencePipeline(BEVFusionInferencePipeline):
    """PyTorch-based BEVFusion pipeline (sparse + dense seams).

    Runs the full model natively, split into the same ``sparse`` / ``dense`` seams the
    ONNX/TensorRT backends use so outputs and the latency breakdown line up across backends.
    The base :meth:`run_model` brackets each seam with CUDA syncs and reports ``sparse_ms`` /
    ``dense_ms``.
    """

    def __init__(self, pytorch_model: torch.nn.Module, device: DeviceSpec) -> None:
        super().__init__(pytorch_model=pytorch_model, backend_type=Backend.PYTORCH, device=device)
        logger.info("BEVFusion PyTorch pipeline initialized (sparse/dense seams)")

    @override
    def run_sparse_encoder(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse branch: voxel encoder (mean pool + sin-cos Fourier) + spconv -> dense BEV."""
        # Model is already loaded in eval mode by ``build_bevfusion_model``; no per-sample eval() here.
        model = self.pytorch_model
        device = self.torch_device

        voxels = voxels.to(device)
        coors = coors.to(device)
        num_points_per_voxel = num_points_per_voxel.to(device)

        with torch.no_grad():
            if coors.shape[1] == 3:
                num_points = coors.shape[0]
                batch_coors = torch.zeros(num_points, 1, device=device, dtype=coors.dtype)
                coors = torch.cat([batch_coors, coors], dim=1).contiguous()

            voxel_features = model.pts_voxel_encoder(voxels, num_points_per_voxel, coors)
            spatial_features = model.pts_middle_encoder(voxel_features, coors, batch_size=1)

        return spatial_features

    @override
    def run_dense(self, bev_features: torch.Tensor) -> List[torch.Tensor]:
        """Dense branch: backbone (SECOND) + neck (SECONDFPN) + bbox_head + scoring."""
        model = self.pytorch_model

        with torch.no_grad():
            backbone_out = bev_features
            if hasattr(model, "pts_backbone") and model.pts_backbone is not None:
                backbone_out = model.pts_backbone(bev_features)

            neck_out = backbone_out
            if hasattr(model, "pts_neck") and model.pts_neck is not None:
                neck_out = model.pts_neck(backbone_out)

            # Match ``BEVFusion.extract_feat``: head ``bev_pos`` is built for
            # ``grid_size // out_size_factor`` (e.g. 180×180) while SECOND/FPN can
            # yield full voxel BEV (e.g. 1440×1440). Skipping this pool causes
            # ``key`` vs ``key_pos`` length mismatch in the transformer decoder.
            align_fn = getattr(model, "_align_lidar_bev_to_head_grid", None)
            if callable(align_fn):
                neck_out = align_fn(neck_out)

            preds = model.bbox_head(neck_out, [])
            preds = preds[0][0]

            # Shared with the ONNX export contract so PyTorch↔ONNX outputs stay bit-identical.
            bbox_pred, score, label_pred = head_dict_to_detection_outputs(preds)

        return [bbox_pred, score, label_pred]
