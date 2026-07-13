"""Shared mmdet3d model-building core for deployment loaders.

Holds the one invariant every project's export loader shares: build the model from an
MMEngine config, load the checkpoint, move it to the target device, put it in eval mode,
and stash the build config on ``model.cfg`` so callers can recover it. Project-specific
concerns (config transforms, module registration, post-load graph fusions) stay in each
project's own loader.
"""

from __future__ import annotations

import copy

import torch
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmengine.runner import load_checkpoint

from deployment.primitives.device import DeviceSpec


def build_mmdet3d_model(
    model_cfg: Config,
    checkpoint_path: str,
    device: DeviceSpec,
) -> torch.nn.Module:
    """Build an mmdet3d model from config, load its checkpoint, and return it in eval mode.

    The project's module variants must already be registered with MMDet3D. Each project's
    loader does this by importing its module packages at import time (``import ...  # noqa:
    F401``), so the registration is in place before this function is called.

    Args:
        model_cfg: MMEngine model configuration whose ``model`` subtree is built.
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: Target device specification.

    Returns:
        The loaded model in eval mode, with ``model.cfg`` set to ``model_cfg`` so callers can
        recover the config it was built from.
    """
    init_default_scope("mmdet3d")

    model = MODELS.build(copy.deepcopy(model_cfg.model))
    torch_device = device.to_torch_device()
    model.to(torch_device)
    load_checkpoint(model, checkpoint_path, map_location=torch_device)
    model.eval()
    model.cfg = model_cfg
    return model
