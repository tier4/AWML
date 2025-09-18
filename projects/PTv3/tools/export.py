import numpy as np
import SparseConvolution  # NOTE(knzo25): do not remove this import, it is needed for onnx export
import spconv.pytorch as spconv
import torch
from engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from engines.train import TRAINERS
from models.scatter.functional import argsort
from models.utils.structure import Point, bit_length_tensor
from torch.nn import functional as F


class WrappedModel(torch.nn.Module):

    def __init__(self, model, cfg):
        super(WrappedModel, self).__init__()
        self.cfg = cfg
        self.model = model.cuda()
        self.model.backbone.forward = self.model.backbone.export_forward

        point_cloud_range = torch.tensor(cfg.point_cloud_range, dtype=torch.float32).cuda()
        voxel_size = cfg.grid_size
        voxel_size = torch.tensor([voxel_size, voxel_size, voxel_size], dtype=torch.float32).cuda()

        self.sparse_shape = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        self.sparse_shape = torch.round(self.sparse_shape).long().cuda()

    def forward(
        self,
        grid_coord,
        feat,
        serialized_depth,
        serialized_code,
    ):

        shape = torch._shape_as_tensor(grid_coord).to(grid_coord.device)

        serialized_order = torch.stack([argsort(code) for code in serialized_code], dim=0)
        serialized_inverse = torch.zeros_like(serialized_order).scatter_(
            dim=1,
            index=serialized_order,
            src=torch.arange(0, serialized_code.shape[1], device=serialized_order.device).repeat(
                serialized_code.shape[0], 1
            ),
        )

        input_dict = {
            "coord": feat[:, :3],
            "grid_coord": grid_coord,
            "offset": shape[:1],
            "feat": feat,
            "serialized_depth": serialized_depth,
            "serialized_code": serialized_code,
            "serialized_order": serialized_order,
            "serialized_inverse": serialized_inverse,
            "sparse_shape": self.sparse_shape,
        }

        output = self.model(input_dict)

        pred_logits = output["seg_logits"]  # (n, k)
        pred_probs = F.softmax(pred_logits, -1)
        pred_label = pred_probs.argmax(-1)

        return pred_label, pred_probs


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    cfg = default_setup(cfg)
    cfg.num_worker = 1
    cfg.num_worker_per_gpu = 1

    # NOTE(knzo25): hacks to allow onnx export
    cfg.model.backbone.shuffle_orders = False
    cfg.model.backbone.order = ["z", "z-trans"]
    cfg.model.backbone.export_mode = True

    runner = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))

    runner.before_train()

    model = WrappedModel(runner.model, cfg)
    model.eval()

    runner.val_loader.prefetch_factor = 1
    data_dict = next(iter(runner.val_loader))

    input_dict = data_dict
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)

    with torch.no_grad():

        depth = bit_length_tensor(
            torch.tensor([(max(cfg.point_cloud_range) - min(cfg.point_cloud_range)) / cfg.grid_size])
        ).cuda()
        point = Point(input_dict)
        point.serialization(
            order=model.model.backbone.order, shuffle_orders=model.model.backbone.shuffle_orders, depth=depth
        )

        input_dict["serialized_depth"] = point["serialized_depth"]
        input_dict["serialized_code"] = point["serialized_code"]
        input_dict.pop("segment")
        input_dict.pop("offset")
        input_dict.pop("coord")

        pred_labels, pred_probs = model(**input_dict)

        np.savez_compressed("ptv3_sample.npz", pred=pred_labels.cpu().numpy(), feat=input_dict["feat"].cpu().numpy())

        export_params = (True,)
        keep_initializers_as_inputs = False
        opset_version = 17
        input_names = ["grid_coord", "feat", "serialized_depth", "serialized_code"]
        output_names = ["pred_labels", "pred_probs"]
        dynamic_axes = {
            "grid_coord": {
                0: "voxels_num",
            },
            "feat": {
                0: "voxels_num",
            },
            "serialized_code": {
                1: "voxels_num",
            },
        }
        torch.onnx.export(
            model,
            input_dict,
            "ptv3.onnx",
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
