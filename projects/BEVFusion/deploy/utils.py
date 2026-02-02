import os
from copy import deepcopy
from dataclasses import dataclass

from mmdeploy.utils import (
    IR,
    Backend,
    get_backend,
    get_dynamic_axes,
    get_ir_config,
    get_onnx_config,
    get_root_logger,
    load_config,
)
from mmdet3d.registry import MODELS
from mmengine.registry import RUNNERS


@dataclass(frozen=True)
class SetupConfigs:
  deploy_cfg_path: str
  model_cfg: dict
  checkpoint_path: str
  device: str
  work_dir: str
  sample_idx: int
  data_preprocessor_cfg: dict


def setup_configs(deploy_cfg_path, model_cfg_path, checkpoint_path, device, work_dir, sample_idx, module):

    os.makedirs(work_dir, exist_ok=True)

    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)
    model_cfg.randomness = dict(seed=0, diff_rank_seed=False, deterministic=False)
    model_cfg.launcher = "none"

    onnx_cfg = get_onnx_config(deploy_cfg)
    input_names = onnx_cfg["input_names"]
    output_names = onnx_cfg["output_names"]

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
      deploy_cfg_path=deploy_cfg_path,
      model_cfg_path=model_cfg_path,
      checkpoint_path=checkpoint_path,
      device=device,
      work_dir=work_dir,
      data_preprocessor_cfg=data_preprocessor_cfg,
    )



def build_model(model_cfg, checkpoint_path, device):

  data_preprocessor = MODELS.build(data_preprocessor_cfg)


  # load a sample
  runner = RUNNERS.build(model_cfg)
  runner.load_or_resume()

    data = runner.test_dataloader.dataset[args.sample_idx]

    # create model an inputs
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    torch_model = task_processor.build_pytorch_model(checkpoint_path)
    data, model_inputs = task_processor.create_input(data, data_preprocessor=data_preprocessor, model=torch_model)

    if isinstance(model_inputs, list) and len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    data_samples = data["data_samples"]
    input_metas = {"data_samples": data_samples, "mode": "predict", "data_preprocessor": data_preprocessor}

    (
        voxels,
        coors,
        num_points_per_voxel,
        points,
        camera_mask,
        imgs,
        lidar2img,
        cam2image,
        camera2lidar,
        geom_feats,
        kept,
        ranks,
        indices,
    ) = model_inputs
