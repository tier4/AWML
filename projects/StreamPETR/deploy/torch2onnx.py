# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This fils is modified from test.py in the original repo.
# Modification: adding 2 proxy class TrtEncoderContainer and TrtPtsHeadContainer
#               and then export these two torch.nn.Module to onnx

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import argparse
import os

import numpy as np
import onnx
import torch
from mmengine import Config
from mmengine.registry import RUNNERS
from mmengine.runner import load_checkpoint
from onnxsim import simplify

from projects.StreamPETR.deploy.containers import (
    TrtEncoderContainer,
    TrtPositionEmbeddingContainer,
    TrtPtsHeadContainer,
)

# torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet benchmark a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--section", help="section can be either extract_img_feat or pts_head_memory")
    parser.add_argument("--checkpoint", help="checkpoint file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # Replace flash attention with normal attention
    cfg.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[0]["type"] = "PETRMultiheadAttention"
    cfg.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[1]["type"] = "PETRMultiheadAttention"

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    runner = RUNNERS.build(cfg)

    model = runner.model

    if args.checkpoint:
        load_checkpoint(
            model,
            args.checkpoint,
        )

    height, width = cfg.ida_aug_conf.final_dim

    if args.section not in ["extract_img_feat", "pts_head_memory", "position_embedding"]:
        raise RuntimeError("unknown section {}".format(args.section))
        exit(-1)

    section = args.section
    if args.section == "extract_img_feat":
        tm = TrtEncoderContainer(model)
        arrs = [
            torch.from_numpy(np.random.uniform(-2, 2, size=(1, cfg.num_cameras, 3, height, width))).float(),
        ]
        input_names = ["img"]
        output_names = ["img_feats"]

    elif args.section == "pts_head_memory":
        tm = TrtPtsHeadContainer(model)
        dmem_init = 1024
        feat_h = int(height / cfg.stride)
        feat_w = int(width / cfg.stride)
        c = int(cfg.model.pts_bbox_head.in_channels)
        print(f"feature size: {c},{feat_h},{feat_w}")
        arrs = [
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, cfg.num_cameras, c, feat_h, feat_w))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, feat_h * feat_w * cfg.num_cameras, c))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, feat_h * feat_w * cfg.num_cameras, 8))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1,))).double(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, 4, 4))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, 4, 4))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, dmem_init, c))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, dmem_init, 3))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, dmem_init, 1))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, dmem_init, 4, 4))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, dmem_init, 2))).float(),
        ]
        input_names = [
            "x",
            "pos_embed",
            "cone",
            "data_timestamp",
            "data_ego_pose",
            "data_ego_pose_inv",
            "pre_memory_embedding",
            "pre_memory_reference_point",
            "pre_memory_timestamp",
            "pre_memory_egopose",
            "pre_memory_velo",
        ]
        output_names = [
            "all_cls_scores",
            "all_bbox_preds",
            "post_memory_embedding",
            "post_memory_reference_point",
            "post_memory_timestamp",
            "post_memory_egopose",
            "post_memory_velo",
            "reference_points",
            "tgt",
            "temp_memory",
            "temp_pos",
            "query_pos",
            "query_pos_in",
            "outs_dec",
        ]
        tm.mod.pts_bbox_head.with_dn = False
    elif args.section == "position_embedding":
        from onnxruntime.tools import pytorch_export_contrib_ops

        pytorch_export_contrib_ops.register()

        feat_h = int(height / cfg.stride)
        feat_w = int(width / cfg.stride)
        c = int(cfg.model.pts_bbox_head.in_channels)

        tm = TrtPositionEmbeddingContainer(model)
        arrs = [
            torch.from_numpy(np.array([height, width, 3])).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, cfg.num_cameras, c, feat_h, feat_w))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, cfg.num_cameras, 4, 4))).float(),
            torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(1, cfg.num_cameras, 4, 4))).float(),
        ]
        input_names = ["img_metas_pad", "img_feats", "intrinsics", "img2lidar"]
        output_names = ["pos_embed", "cone"]

    tm.float()
    tm.cpu()
    tm.eval()
    tm.training = False
    args = tuple(arrs)
    with torch.no_grad():
        torch.onnx.export(
            tm,
            args,
            os.path.join(cfg.work_dir, "{}.onnx".format(section)),
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            verbose=True,
        )

    filename = os.path.join(cfg.work_dir, "{}.onnx".format(section))
    onnx_model = onnx.load(filename)
    onnx_model_simp, check = simplify(onnx_model)
    onnx.save(onnx_model_simp, os.path.join(cfg.work_dir, "simplify_{}.onnx".format(section)))

    print("Completed...")


if __name__ == "__main__":
    main()
