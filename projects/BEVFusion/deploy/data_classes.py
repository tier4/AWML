# Copyright (c) OpenMMLab. All rights reserved.

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SetupConfigs:
    """Setup configurations for the model."""

    deploy_cfg: dict
    model_cfg: dict
    checkpoint_path: str
    device: str
    data_preprocessor_cfg: dict
    sample_idx: int
    module: str
    onnx_cfg: dict
    work_dir: str


@dataclass(frozen=True)
class ModelInputs:
    """Model inputs for the model."""

    voxels: torch.Tensor
    coors: torch.Tensor
    num_points_per_voxel: torch.Tensor
    points: torch.Tensor
    camera_mask: torch.Tensor
    imgs: torch.Tensor
    lidar2img: torch.Tensor
    cam2image: torch.Tensor
    camera2lidar: torch.Tensor
    geom_feats: torch.Tensor
    kept: torch.Tensor
    ranks: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class ModelData:
    """Model data for the model."""

    model_inputs: ModelInputs
    torch_model: torch.nn.Module
    input_metas: dict


@dataclass(frozen=True)
class BackendConfigs:
    """Backend configurations for the model."""

    type: str
    optimize: bool


@dataclass(frozen=True)
class BuilderData:
    """Builder data for the model."""

    model_data: ModelData
    ir_configs: dict
    context_info: dict
    patched_model: torch.nn.Module
