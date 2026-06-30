from dataclasses import fields

import numpy as np
import spconv.pytorch as spconv
import torch
from engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from engines.train import TRAINERS
from models.point_transformer_v3.point_transformer_v3m1_base import (
    SerializedPoolingMeta,
    build_serialized_pooling_meta,
)
from models.scatter.functional import argsort
from models.utils.structure import Point, bit_length_tensor
from torch.nn import functional as F

# NOTE: keep this import last; it overrides sparse conv registration for export.
import SparseConvolution  # isort: skip

SERIALIZED_POOLING_FIELDS = tuple(f.name for f in fields(SerializedPoolingMeta))


def build_serialized_pooling_metadata(grid_coord, serialized_code, serialized_order, strides):
    """Build the pooling metadata for every encoder stage by chaining the stages together."""
    metadata = []
    for stride in strides:
        meta, serialized_code = build_serialized_pooling_meta(grid_coord, serialized_code, serialized_order, stride)
        metadata.append(meta)
        grid_coord, serialized_order = meta.grid_coord, meta.serialized_order
    return metadata


def _serialized_pooling_dynamic_axis(stage_index, field):
    """Return the single dynamic ONNX axis for one metadata field of one pooling stage."""
    in_voxels = f"serialized_pooling_{stage_index}_in_voxels"
    out_voxels = f"serialized_pooling_{stage_index}_out_voxels"
    match field:
        case "indices" | "cluster":
            return {0: in_voxels}
        case "indptr":  # holds M + 1 entries, so it gets its own symbolic dim
            return {0: f"{out_voxels}_plus_one"}
        case "head_indices" | "grid_coord":
            return {0: out_voxels}
        case "serialized_order" | "serialized_inverse":
            return {1: out_voxels}
        case _:
            raise AssertionError(f"unexpected serialized-pooling field: {field!r}")


def flatten_serialized_pooling_inputs(metadata):
    """Flatten per-stage metadata into ordered ONNX inputs, input names, and dynamic axes."""
    flat_inputs, input_names, dynamic_axes = [], [], {}
    for stage_index, meta in enumerate(metadata):
        for field in SERIALIZED_POOLING_FIELDS:
            name = f"serialized_pooling_{stage_index}_{field}"
            flat_inputs.append(getattr(meta, field))
            input_names.append(name)
            dynamic_axes[name] = _serialized_pooling_dynamic_axis(stage_index, field)
    return flat_inputs, input_names, dynamic_axes


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
        *serialized_pooling_inputs,
    ):
        num_fields = len(SERIALIZED_POOLING_FIELDS)
        assert len(serialized_pooling_inputs) % num_fields == 0

        shape = torch._shape_as_tensor(grid_coord).to(grid_coord.device)

        serialized_order = torch.stack([argsort(code) for code in serialized_code], dim=0)
        serialized_inverse = torch.zeros_like(serialized_order).scatter_(
            dim=1,
            index=serialized_order,
            src=torch.arange(0, serialized_code.shape[1], device=serialized_order.device).repeat(
                serialized_code.shape[0], 1
            ),
        )

        serialized_pooling = [
            SerializedPoolingMeta(*serialized_pooling_inputs[i : i + num_fields])
            for i in range(0, len(serialized_pooling_inputs), num_fields)
        ]

        input_dict = {
            "coord": feat[:, :3],
            "grid_coord": grid_coord,
            "offset": shape[:1],
            "feat": feat,
            "serialized_depth": serialized_depth,
            "serialized_code": serialized_code,
            "serialized_order": serialized_order,
            "serialized_inverse": serialized_inverse,
            "serialized_pooling": serialized_pooling,
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
        serialized_pooling = build_serialized_pooling_metadata(
            point["grid_coord"],
            point["serialized_code"],
            point["serialized_order"],
            cfg.model.backbone.stride,
        )
        serialized_pooling_inputs, serialized_pooling_input_names, serialized_pooling_dynamic_axes = (
            flatten_serialized_pooling_inputs(serialized_pooling)
        )

        input_dict["serialized_depth"] = point["serialized_depth"]
        input_dict["serialized_code"] = point["serialized_code"]
        input_dict.pop("segment")
        input_dict.pop("offset")
        input_dict.pop("coord")

        model_inputs = (
            input_dict["grid_coord"],
            input_dict["feat"],
            input_dict["serialized_depth"],
            input_dict["serialized_code"],
            *serialized_pooling_inputs,
        )

        pred_labels, pred_probs = model(*model_inputs)

        np.savez_compressed("ptv3_sample.npz", pred=pred_labels.cpu().numpy(), feat=input_dict["feat"].cpu().numpy())

        export_params = (True,)
        keep_initializers_as_inputs = False
        opset_version = 17
        input_names = [
            "grid_coord",
            "feat",
            "serialized_depth",
            "serialized_code",
            *serialized_pooling_input_names,
        ]
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
        dynamic_axes.update(serialized_pooling_dynamic_axes)
        torch.onnx.export(
            model,
            model_inputs,
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
