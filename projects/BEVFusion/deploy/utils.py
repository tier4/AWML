# Copyright (c) OpenMMLab. All rights reserved.

import os
from copy import deepcopy

from data_classes import SetupConfigs
from mmdeploy.utils import (
    get_onnx_config,
    load_config,
)


def setup_configs(
    deploy_cfg_path: str,
    model_cfg_path: str,
    checkpoint_path: str,
    device: str,
    work_dir: str,
    sample_idx: int,
    module: str,
) -> SetupConfigs:
    """
    Setup configuration for the model.

    Args:
        deploy_cfg_path: Path to the deploy config file.
        model_cfg_path: Path to the model config file.
        checkpoint_path: Path to the checkpoint file.
        device: Device to use for the model.
        work_dir: Directory to save the model.
        sample_idx: Index of the sample to use for the model.
        module: Module to export.
    """
    os.makedirs(work_dir, exist_ok=True)
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
    model_cfg.randomness = dict(seed=0, diff_rank_seed=False, deterministic=False)
    model_cfg.launcher = "none"

    onnx_cfg = get_onnx_config(deploy_cfg)
    input_names = onnx_cfg["input_names"]

    extract_pts_inputs = True if "points" in input_names or "voxels" in input_names else False
    data_preprocessor_cfg = deepcopy(model_cfg.model.data_preprocessor)

    # TODO(KokSeang): Move out from data_preprocessor
    voxelize_cfg = deepcopy(model_cfg.get("voxelize_cfg", None))

    if extract_pts_inputs and voxelize_cfg is None:
        # TODO(KokSeang): Remove this
        # Default voxelize_layer
        voxelize_cfg = dict(
            max_num_points=10,
            voxel_size=[0.17, 0.17, 0.2],
            point_cloud_range=[-122.4, -122.4, -3.0, 122.4, 122.4, 5.0],
            max_voxels=[120000, 160000],
            deterministic=True,
        )

    if voxelize_cfg is not None:
        voxelize_cfg.pop("voxelize_reduce", None)
        data_preprocessor_cfg["voxel_layer"] = voxelize_cfg
        data_preprocessor_cfg.voxel = True

    # load a sample
    if "work_dir" not in model_cfg:
        model_cfg["work_dir"] = work_dir

    return SetupConfigs(
        deploy_cfg=deploy_cfg,
        model_cfg=model_cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        data_preprocessor_cfg=data_preprocessor_cfg,
        sample_idx=sample_idx,
        module=module,
        onnx_cfg=onnx_cfg,
        work_dir=work_dir,
        extract_pts_inputs=extract_pts_inputs,
    )
