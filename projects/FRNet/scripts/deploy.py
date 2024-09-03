import argparse
import os

import torch

from mmengine.config import Config
from mmengine.runner import Runner


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Exiting...")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy a model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file path",
        required=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/projects/FRNet/configs/nuscenes/frnet_1xb4_nus-seg.py",
        help="Config file path",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="/workspace/work_dirs/deploy",
        help="The directory to save logs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint
    output_path = os.path.join(cfg.work_dir, "frnet.onnx")

    runner = Runner.from_cfg(cfg)
    runner.model.eval()

    batch_inputs_dict = {
        "voxels": {
            "voxels": torch.randn(1 << 16, 4, dtype=torch.float32, device="cuda"),
            "coors": torch.zeros(1 << 16, 3, dtype=torch.int64, device="cuda"),
        }
    }

    torch.onnx.export(
        runner.model,
        batch_inputs_dict,
        output_path,
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=17,
        do_constant_folding=True,
        input_names=["voxels", "coors"],
        output_names=["voxels0", "coors0", "voxel_feats0", "voxel_feats1", "voxel_feats2", "voxel_feats3", "voxel_feats4",
                      "voxel_coors0", "point_feats0", "point_feats1", "point_feats2", "point_feats3",
                      "point_feats_backbone0", "point_feats_backbone1", "point_feats_backbone2", "point_feats_backbone3", "point_feats_backbone4",
                      "seg_logit0"],
        dynamic_axes={
            "voxels": {
                0: "num_points"
            },
            "coors": {
                0: "num_points"
            }
        }
    )

    print(f"ONNX model saved to {output_path}.")


if __name__ == "__main__":
    main()
