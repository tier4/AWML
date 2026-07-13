"""BEVFusion model loading utilities for deployment."""

from __future__ import annotations

import logging

import torch
from mmengine.config import Config

# Imported for their side effect: registering BEVFusion and SparseConvolution modules into the
# MMDet3D registries so ``MODELS.build`` can resolve them during export.
import projects.BEVFusion.bevfusion  # noqa: F401
import projects.SparseConvolution  # noqa: F401
from deployment.io.mmdet3d_model import build_mmdet3d_model
from deployment.primitives.device import DeviceSpec
from deployment.projects.bevfusion_l.export.spconv_bn_fusion import fuse_spconv_bn_in_encoder

logger = logging.getLogger(__name__)


def _require_lidar_only_bevfusion(model: torch.nn.Module) -> None:
    """Assert the loaded checkpoint is a LiDAR-only BEVFusion model.

    The ``bevfusion_l`` bundle only deploys the LiDAR path (voxels -> sparse encoder -> dense head);
    it has no camera/fusion export. A camera (``bevfusion_c``) or fusion (``bevfusion_cl``)
    checkpoint would trace a graph this bundle cannot serve, so fail loud once here at load with a
    clear message rather than deep inside ONNX export. ``pts_middle_encoder`` is the sparse encoder
    every export path and the PyTorch backend depend on, so its absence is also caught here.
    """
    if getattr(model, "fusion_layer", None) is not None:
        raise RuntimeError(
            "bevfusion_l deploys LiDAR-only BEVFusion, but the loaded checkpoint has a fusion_layer. "
            "Use a LiDAR-only checkpoint (a camera/fusion model needs a dedicated bevfusion_c / "
            "bevfusion_cl bundle)."
        )
    if getattr(model, "img_backbone", None) is not None:
        raise RuntimeError(
            "bevfusion_l deploys LiDAR-only BEVFusion, but the loaded checkpoint has an img_backbone. "
            "Use a LiDAR-only checkpoint (a camera/fusion model needs a dedicated bevfusion_c / "
            "bevfusion_cl bundle)."
        )
    if getattr(model, "pts_middle_encoder", None) is None:
        raise RuntimeError(
            "bevfusion_l requires a sparse pts_middle_encoder (LiDAR BEVFusion), but the loaded "
            "checkpoint has none."
        )


def build_bevfusion_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
    *,
    fuse_spconv_bn: bool = False,
) -> torch.nn.Module:
    """Build a BEVFusion model from config and load checkpoint weights.

    Args:
        model_cfg: MMEngine model configuration.
        checkpoint_path: Path to .pth checkpoint file.
        device: Target device.
        fuse_spconv_bn: If True, fuse each ``SparseConvolution`` + ``BatchNorm1d`` pair in
            ``pts_middle_encoder`` after ``load_checkpoint`` (eval-mode Conv-BN fold, a graph
            optimization for the sparse ONNX export).

    Returns:
        Loaded and eval-mode BEVFusion model.

    Raises:
        RuntimeError: If the checkpoint is not a LiDAR-only BEVFusion model (see
            :func:`_require_lidar_only_bevfusion`).
    """
    model = build_mmdet3d_model(model_cfg, checkpoint_path, device)
    _require_lidar_only_bevfusion(model)

    if fuse_spconv_bn:
        encoder = getattr(model, "pts_middle_encoder", None)
        if encoder is not None:
            count = fuse_spconv_bn_in_encoder(encoder)
            logger.info("Fused %d SparseConv-BN pair(s) in pts_middle_encoder", count)

    return model
