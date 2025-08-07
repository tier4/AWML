# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import onnx
import torch
from containers import TrtBevFusionImageBackboneContainer, TrtBevFusionMainContainer
from mmdeploy.apis import build_task_processor
from mmdeploy.apis.onnx.passes import optimize_onnx
from mmdeploy.core import RewriterContext, patch_model
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
from mmengine.runner import load_checkpoint
from torch.multiprocessing import set_start_method


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to onnx.")
    parser.add_argument("deploy_cfg", help="deploy config path")
    parser.add_argument("model_cfg", help="model config path")
    parser.add_argument("checkpoint", help="model checkpoint path")
    parser.add_argument("--work-dir", default=os.getcwd(), help="the dir to save logs and models")
    parser.add_argument("--device", help="device used for conversion", default="cpu")
    parser.add_argument("--log-level", help="set log level", default="INFO", choices=list(logging._nameToLevel.keys()))
    parser.add_argument("--sample_idx", type=int, default=0, help="sample index to use during export")
    parser.add_argument(
        "--module",
        help="module to export",
        required=True,
        default="main_body",
        choices=["main_body", "image_backbone"],
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_start_method("spawn", force=True)
    logger = get_root_logger()
    log_level = logging.getLevelName(args.log_level)
    logger.setLevel(log_level)

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint
    device = args.device
    work_dir = args.work_dir

    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    model_cfg.randomness = dict(seed=0, diff_rank_seed=False, deterministic=False)
    model_cfg.launcher = "none"

    data_preprocessor_cfg = deepcopy(model_cfg.model.data_preprocessor)

    voxelize_cfg = data_preprocessor_cfg.pop("voxelize_cfg")
    voxelize_cfg.pop("voxelize_reduce")
    data_preprocessor_cfg["voxel_layer"] = voxelize_cfg
    data_preprocessor_cfg.voxel = True

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
        lidar2image,
        cam2image,
        camera2lidar,
        geom_feats,
        kept,
        ranks,
        indices,
    ) = model_inputs

    # export to onnx
    context_info = dict()
    context_info["deploy_cfg"] = deploy_cfg
    output_prefix = osp.join(work_dir, osp.splitext(osp.basename(deploy_cfg.onnx_config.save_file))[0])
    os.makedirs(work_dir, exist_ok=True)
    backend = get_backend(deploy_cfg).value

    onnx_cfg = get_onnx_config(deploy_cfg)
    opset_version = onnx_cfg.get("opset_version", 11)

    input_names = onnx_cfg["input_names"]
    output_names = onnx_cfg["output_names"]
    axis_names = input_names + output_names
    dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)
    verbose = not onnx_cfg.get("strip_doc_string", True) or onnx_cfg.get("verbose", False)
    keep_initializers_as_inputs = onnx_cfg.get("keep_initializers_as_inputs", True)
    optimize = onnx_cfg.get("optimize", False)
    if backend == Backend.NCNN.value:
        """NCNN backend needs a precise blob counts, while using onnx optimizer
        will merge duplicate initilizers without reference count."""
        optimize = False

    output_path = output_prefix + ".onnx"

    logger = get_root_logger()
    logger.info(f"Export PyTorch model to ONNX: {output_path}.")

    def _add_or_update(cfg: dict, key: str, val: Any):
        if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
            cfg[key].update(val)
        else:
            cfg[key] = val

    ir_config = dict(
        type="onnx",
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
    )
    _add_or_update(deploy_cfg, "ir_config", ir_config)
    ir = IR.get(get_ir_config(deploy_cfg)["type"])
    if isinstance(backend, Backend):
        backend = backend.value
    backend_config = dict(type=backend)
    _add_or_update(deploy_cfg, "backend_config", backend_config)

    context_info["cfg"] = deploy_cfg
    context_info["ir"] = ir
    if "backend" not in context_info:
        context_info["backend"] = backend
    if "opset" not in context_info:
        context_info["opset"] = opset_version

    # patch model
    patched_model = patch_model(torch_model, cfg=deploy_cfg, backend=backend, ir=ir)
    patched_model.eval()
    patched_model.to(device)
    if "onnx_custom_passes" not in context_info:
        onnx_custom_passes = optimize_onnx if optimize else None
        context_info["onnx_custom_passes"] = onnx_custom_passes
    with RewriterContext(**context_info), torch.no_grad():
        image_feats = None

        if "img_backbone" in model_cfg.model:
            img_aug_matrix = imgs.new_tensor(np.stack(data_samples[0].img_aug_matrix))
            lidar_aug_matrix = torch.eye(4).to(imgs.device)
            images_mean = data_preprocessor.mean.to(device)
            images_std = data_preprocessor.std.to(device)
            image_backbone_container = TrtBevFusionImageBackboneContainer(patched_model, images_mean, images_std)
            model_inputs = (
                imgs.unsqueeze(0).to(device).float(),
                points.unsqueeze(0).to(device).float(),
                lidar2image.unsqueeze(0).to(device).float(),
                cam2image.unsqueeze(0).to(device).float(),
                torch.inverse(cam2image).unsqueeze(0).to(device).float(),
                camera2lidar.unsqueeze(0).to(device).float(),
                img_aug_matrix.unsqueeze(0).to(device).float(),
                imgs.new_tensor(np.stack([np.linalg.inv(x) for x in data_samples[0].img_aug_matrix]))
                .unsqueeze(0)
                .to(device)
                .float(),
                lidar_aug_matrix.unsqueeze(0).to(device).float(),
                torch.inverse(lidar_aug_matrix).unsqueeze(0).to(device).float(),
                geom_feats.to(device).float(),
                kept.to(device),
                ranks.to(device).long(),
                indices.to(device).long(),
            )
            if args.module == "image_backbone":
                return_value = torch.onnx.export(
                    image_backbone_container,
                    model_inputs,
                    output_path,
                    export_params=True,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=opset_version,
                    dynamic_axes=dynamic_axes,
                    keep_initializers_as_inputs=keep_initializers_as_inputs,
                    verbose=verbose,
                )
            else:
                image_feats = image_backbone_container(*model_inputs)

        if args.module == "main_body":
            main_container = TrtBevFusionMainContainer(patched_model)
            model_inputs = (
                voxels.to(device),
                coors.to(device),
                num_points_per_voxel.to(device),
            )
            if image_feats is not None:
                model_inputs += (image_feats,)
            torch.onnx.export(
                main_container,
                model_inputs,
                output_path,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                dynamic_axes=dynamic_axes,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
                verbose=verbose,
            )
    logger.info(f"ONNX exported to {output_path}")

    logger.info("Attempting to fix the graph (TopK's K becoming a tensor)")

    import onnx_graphsurgeon as gs

    model = onnx.load(output_path)
    graph = gs.import_onnx(model)

    # Fix TopK
    topk_nodes = [node for node in graph.nodes if node.op == "TopK"]
    assert len(topk_nodes) == 1
    topk = topk_nodes[0]
    k = model_cfg.num_proposals
    topk.inputs[1] = gs.Constant("K", values=np.array([k], dtype=np.int64))
    topk.outputs[0].shape = [1, k]
    topk.outputs[0].dtype = topk.inputs[0].dtype if topk.inputs[0].dtype else np.float32
    topk.outputs[1].shape = [1, k]
    topk.outputs[1].dtype = np.int64

    graph.cleanup().toposort()
    output_path = output_path.replace(".onnx", "_fixed.onnx")
    onnx.save_model(gs.export_onnx(graph), output_path)

    logger.info(f"(Fixed) ONNX exported to {output_path}")
