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
from models.point_transformer_v3.point_transformer_v3m1_base import SerializedPoolingMeta
from models.scatter.functional import argsort
from models.utils.structure import Point, bit_length_tensor
from torch.nn import functional as F

# NOTE: keep this import last; it overrides sparse conv registration for export.
import SparseConvolution  # isort: skip

# Per-stage ONNX input field order, kept in sync with the model-side dataclass.
SERIALIZED_POOLING_FIELDS = tuple(f.name for f in fields(SerializedPoolingMeta))


def pooling_depth(stride):
    depth = 0
    value = int(stride)
    while value > 1:
        depth += 1
        value >>= 1
    return depth


def build_serialized_pooling_metadata(grid_coord, serialized_code, serialized_order, stride):
    depth = pooling_depth(stride)
    pooled_code = serialized_code >> (depth * 3)

    indices = serialized_order[0]
    sorted_code = pooled_code[0].index_select(0, indices)
    run_start = torch.cat(
        [
            torch.ones_like(sorted_code[:1], dtype=torch.bool),
            sorted_code[1:] != sorted_code[:-1],
        ],
        dim=0,
    )
    run_start_indices = torch.nonzero(run_start, as_tuple=False).flatten()
    input_count = torch._shape_as_tensor(indices).to(indices.device)[:1]
    indptr = torch.cat([run_start_indices, input_count], dim=0)

    cluster_sorted = torch.cumsum(run_start.to(dtype=indices.dtype), dim=0) - 1
    cluster = torch.zeros_like(cluster_sorted)
    cluster.scatter_(0, indices, cluster_sorted)

    head_indices = indices.index_select(0, indptr[:-1])
    next_grid_coord = grid_coord.index_select(0, head_indices) >> depth
    next_serialized_code = pooled_code.index_select(1, head_indices)
    next_serialized_order = torch.stack([argsort(code) for code in next_serialized_code], dim=0)
    next_serialized_inverse = torch.zeros_like(next_serialized_order).scatter_(
        dim=1,
        index=next_serialized_order,
        src=torch.arange(
            0,
            next_serialized_code.shape[1],
            device=next_serialized_order.device,
        ).repeat(next_serialized_code.shape[0], 1),
    )

    return {
        "indices": indices,
        "indptr": indptr,
        "cluster": cluster,
        "head_indices": head_indices,
        "grid_coord": next_grid_coord,
        "serialized_code": next_serialized_code,
        "serialized_order": next_serialized_order,
        "serialized_inverse": next_serialized_inverse,
    }


def build_serialized_pooling_inputs(grid_coord, serialized_code, serialized_order, strides):
    """Build sample tensors for the ONNX/inference serialized-pooling contract.

    These tensors are not constants in the exported graph. They are sample inputs used by
    torch.onnx.export to define the interface that Autoware preprocessing fills at runtime.
    """

    metadata_by_stage = []
    current_grid_coord = grid_coord
    current_code = serialized_code
    current_order = serialized_order
    for stride in strides:
        metadata = build_serialized_pooling_metadata(current_grid_coord, current_code, current_order, stride)
        metadata_by_stage.append(metadata)
        current_grid_coord = metadata["grid_coord"]
        current_code = metadata["serialized_code"]
        current_order = metadata["serialized_order"]

    return metadata_by_stage


def flatten_serialized_pooling_inputs(metadata_by_stage):
    flat_inputs = []
    input_names = []
    dynamic_axes = {}
    for stage_index, metadata in enumerate(metadata_by_stage):
        prefix = f"serialized_pooling_{stage_index}_"
        for field in SERIALIZED_POOLING_FIELDS:
            name = prefix + field
            flat_inputs.append(metadata[field])
            input_names.append(name)
            if field in {"grid_coord", "serialized_order", "serialized_inverse"}:
                axis = 0 if field == "grid_coord" else 1
                dynamic_axes[name] = {axis: f"serialized_pooling_{stage_index}_out_voxels"}
            elif field in {"head_indices", "indptr"}:
                dynamic_axes[name] = {0: f"serialized_pooling_{stage_index}_out_voxels"}
            else:
                dynamic_axes[name] = {0: f"serialized_pooling_{stage_index}_in_voxels"}
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
        num_pooling_fields = len(SERIALIZED_POOLING_FIELDS)
        assert len(serialized_pooling_inputs) % num_pooling_fields == 0

        shape = torch._shape_as_tensor(grid_coord).to(grid_coord.device)

        serialized_order = torch.stack([argsort(code) for code in serialized_code], dim=0)
        serialized_inverse = torch.zeros_like(serialized_order).scatter_(
            dim=1,
            index=serialized_order,
            src=torch.arange(0, serialized_code.shape[1], device=serialized_order.device).repeat(
                serialized_code.shape[0], 1
            ),
        )

        serialized_pooling = []
        for stage_index in range(len(serialized_pooling_inputs) // num_pooling_fields):
            stage_values = serialized_pooling_inputs[
                stage_index * num_pooling_fields : (stage_index + 1) * num_pooling_fields
            ]
            serialized_pooling.append(SerializedPoolingMeta(**dict(zip(SERIALIZED_POOLING_FIELDS, stage_values))))

        input_dict = {
            "coord": feat[:, :3],
            "grid_coord": grid_coord,
            "offset": shape[:1],
            "feat": feat,
            "serialized_depth": serialized_depth,
            "serialized_code": serialized_code,
            "serialized_order": serialized_order,
            "serialized_inverse": serialized_inverse,
            # List of SerializedPoolingMeta dataclasses; addict stores these verbatim (only plain
            # dicts get auto-converted to Point, which would recurse on the missing "coord").
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
        serialized_pooling = build_serialized_pooling_inputs(
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
