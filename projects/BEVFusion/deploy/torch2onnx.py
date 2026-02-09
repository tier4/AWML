# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os

from exporter import Torch2OnnxExporter
from torch.multiprocessing import set_start_method
from utils import setup_configs


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
        choices=["main_body", "image_backbone", "camera_bev_only_network"],
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_start_method("spawn", force=True)
    setup_config = setup_configs(
        args.deploy_cfg,
        args.model_cfg,
        args.checkpoint,
        args.device,
        args.work_dir,
        args.sample_idx,
        args.module,
    )
    # Build the exporter
    exporter = Torch2OnnxExporter(setup_config, args.log_level)

    # Export the model
    exporter.export()
