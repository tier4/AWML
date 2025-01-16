"""
Script to export CenterPoint to onnx/torchscript
"""

import argparse
import logging
import os
from pathlib import Path

from projects.CenterPoint.runners.deployment_runner import DeploymentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export CenterPoint model to backends.",
    )
    parser.add_argument(
        "model_cfg_path",
        help="model config path",
    )
    parser.add_argument(
        "checkpoint",
        help="model checkpoint path",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="the dir to save logs and models",
    )
    parser.add_argument(
        "--log-level",
        help="set log level",
        default="INFO",
        choices=list(logging._nameToLevel.keys()),
    )
    parser.add_argument("--onnx_opset_version", type=int, default=13, help="onnx opset version")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Set running device!",
    )
    parser.add_argument(
        "--replace_onnx_models",
        action="store_true",
        help="Set False to disable replacement of model by ONNX model, for example, CenterHead -> CenterHeadONNX",
    )
    parser.add_argument(
        "--rot_y_axis_reference",
        action="store_true",
        help="Set True to output rotation in y-axis clockwise in CenterHeadONNX",
    )
    args = parser.parse_args()
    return args


def build_deploy_runner(args) -> DeploymentRunner:
    """Build a DeployRunner."""
    model_cfg_path = args.model_cfg_path
    checkpoint_path = args.checkpoint
    experiment_name = Path(model_cfg_path).stem
    work_dir = (
        Path(os.getcwd()) / "work_dirs" / "deployment" / experiment_name if not args.work_dir else Path(args.work_dir)
    )

    deployment_runner = DeploymentRunner(
        experiment_name=experiment_name,
        model_cfg_path=model_cfg_path,
        checkpoint_path=checkpoint_path,
        work_dir=work_dir,
        replace_onnx_models=args.replace_onnx_models,
        device=args.device,
        rot_y_axis_reference=args.rot_y_axis_reference,
        onnx_opset_version=args.onnx_opset_version,
    )
    return deployment_runner


if __name__ == "__main__":
    """Launch a DeployRunner."""
    args = parse_args()

    # Build DeploymentRunner
    deployment_runner = build_deploy_runner(args=args)

    # Start running DeploymentRunner
    deployment_runner.run()
