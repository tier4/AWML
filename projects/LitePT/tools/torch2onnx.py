from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import SparseConvolution  # noqa do not remove this line, it is required for onnx export
from litept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from litept.engines.train import TRAINERS
from litept.models.utils.structure import Point, bit_length_tensor
from litept.utils.logger import get_root_logger
from litept.utils.visualization import get_segmentation_colors, visualize_point_cloud

logger = get_root_logger()


class LitePTONNX(nn.Module):
    """
    Wrapper to present a stable ONNX I/O signature.

    Inputs:
      - grid_coord:  (N, 3) int32/int64
      - feat:        (N, F) float32/float16
      - serialized_depth: (N,) int32/int64
      - serialized_code: (N,) int32/int64

    Output:
      - pred_label:    (N_down,) int64
      - pred_probs:    (N_down, C) float32/float16
    """

    def __init__(self, cfg, model) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model

        point_cloud_range = torch.tensor(cfg.point_cloud_range, dtype=torch.float32).cuda()
        voxel_size = cfg.grid_size
        voxel_size = torch.tensor([voxel_size, voxel_size, voxel_size], dtype=torch.float32).cuda()

        self.sparse_shape = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        self.sparse_shape = torch.round(self.sparse_shape).long().cuda()

    def forward(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = torch._shape_as_tensor(grid_coord).to(grid_coord.device)

        input_dict = {
            "coord": feat[:, :3],
            "grid_coord": grid_coord,
            "offset": shape[:1],
            "feat": feat,
            "sparse_shape": self.sparse_shape,
        }
        output = self.model(input_dict)

        pred_logits = output["seg_logits"]  # (n, k)
        pred_score = F.softmax(pred_logits, -1)
        pred_label = pred_score.argmax(-1)

        return pred_label, pred_score


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    cfg = default_setup(cfg)
    cfg.num_worker = 1
    cfg.num_worker_per_gpu = 1

    # NOTE: hacks to allow onnx export
    cfg.model.backbone.shuffle_orders = False
    cfg.model.backbone.order = ["z", "z-trans"]
    cfg.model.backbone.export_mode = True

    runner = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    runner.before_train()

    model = LitePTONNX(cfg, runner.model)
    model.eval()

    runner.val_loader.prefetch_factor = 1
    data_dict = next(iter(runner.val_loader))

    input_dict = data_dict
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)

    with torch.no_grad():
        depth = bit_length_tensor(
            torch.tensor(
                [(max(cfg.point_cloud_range) - min(cfg.point_cloud_range)) / cfg.grid_size]
            )
        ).cuda()
        point = Point(input_dict)
        point.serialization(
            order=model.model.backbone.order,
            depth=depth,
            shuffle_orders=model.model.backbone.shuffle_orders,
        )

        input_dict["serialized_depth"] = point["serialized_depth"]
        input_dict["serialized_code"] = point["serialized_code"]
        input_dict.pop("coord")
        input_dict.pop("segment")
        input_dict.pop("origin_segment")
        input_dict.pop("inverse")
        input_dict.pop("offset")

        pred_labels, pred_probs = model(input_dict["grid_coord"], input_dict["feat"])

        sample_filepath = "litept_sample.npz"
        np.savez_compressed(
            sample_filepath,
            pred=pred_labels.cpu().numpy(),
            feat=input_dict["feat"].cpu().numpy(),
        )
        if cfg.get("show", False):
            # visualize sample
            coords, colors = get_segmentation_colors(
                pred_labels.cpu().numpy(), sample_filepath, cfg.class_colors, logger
            )
            visualize_point_cloud(coords, colors, "Predictions")

        export_params = True
        keep_initializers_as_inputs = False
        opset_version = 17
        input_names = ["grid_coord", "feat"]
        output_names = ["pred_label", "pred_score"]
        dynamic_axes = {
            "grid_coord": {
                0: "voxels_num",
            },
            "feat": {
                0: "voxels_num",
            },
        }
        torch.onnx.export(
            model,
            (
                input_dict["grid_coord"],
                input_dict["feat"],
            ),
            "litept.onnx",
            export_params=export_params,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            verbose=False,
            do_constant_folding=False,
        )

    print("Exported to ONNX format successfully.")


if __name__ == "__main__":
    main()
