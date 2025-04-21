from datasets import collate_fn

from engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from engines.train import TRAINERS
from engines.test import TESTERS
from engines.launch import launch

from models.utils.structure import Point

import torch
from torch.nn import functional as F
import numpy as np

from typing import List, Dict, Any

import onnx

import spconv.pytorch as spconv
import SparseConvolution


class WrappedModel(torch.nn.Module):

    def __init__(self, model, cfg):
        super(WrappedModel, self).__init__()
        self.cfg = cfg
        self.model = model.cuda()
        #self.model.forward = self.model.export_forward
        self.model.backbone.forward = self.model.backbone.export_forward

        point_cloud_range = torch.tensor(cfg.point_cloud_range, dtype=torch.float32).cuda()
        voxel_size = cfg.grid_size
        voxel_size = torch.tensor([voxel_size, voxel_size, voxel_size], dtype=torch.float32).cuda()

        self.sparse_shape = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        self.sparse_shape = torch.round(self.sparse_shape).long().cuda()

        #c = floor((points[i][j] - coors_range[j]) / voxel_size[j]);

    def forward(
        self,
        coord,
        grid_coord,
        offset,
        feat,
        serialized_depth,
        serialized_code,
        serialized_order,
        serialized_inverse,
    ):

        input_dict = {
            "coord": coord,
            "grid_coord": grid_coord,
            "offset": offset,
            "feat": feat,
            "serialized_depth": serialized_depth,
            "serialized_code": serialized_code,
            "serialized_order": serialized_order,
            "serialized_inverse": serialized_inverse,
            "sparse_shape": self.sparse_shape,
        }

        output = self.model(input_dict)

        pred_part = output["seg_logits"]  # (n, k)
        pred_part = F.softmax(pred_part, -1)
        pred_part = pred_part.argmax(-1)

        return pred_part


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    cfg = default_setup(cfg)
    cfg.num_worker = 1
    cfg.num_worker_per_gpu = 1

    # NOTE(knzo25): hacks to allow onnx export
    cfg.model.backbone.shuffle_orders = False
    cfg.model.backbone.order = ["z", "z-trans"]

    runner = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))

    runner.before_train()

    model = WrappedModel(runner.model, cfg)
    model.eval()

    data_dict = next(iter(runner.val_loader))

    input_dict = data_dict
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)

    with torch.no_grad():

        point = Point(input_dict)
        point.serialization(order=model.model.backbone.order,
                            shuffle_orders=model.model.backbone.shuffle_orders)

        input_dict["serialized_depth"] = point["serialized_depth"]
        input_dict["serialized_code"] = point["serialized_code"]
        input_dict["serialized_order"] = point["serialized_order"]
        input_dict["serialized_inverse"] = point["serialized_inverse"]
        input_dict.pop("segment")

        pred = model(**input_dict)

        np.savez_compressed("test.npz",
                            pred=pred.cpu().numpy(),
                            feat=input_dict["feat"].cpu().numpy())

        output_path = "test.onnx"

        export_params = True,
        keep_initializers_as_inputs = False
        opset_version = 17
        input_names = [
            "coord", "grid_coord", "offset", "feat", "serialized_depth",
            "serialized_code", "serialized_order", "serialized_inverse"
        ]
        output_names = ["seg"]
        dynamic_axes = {
            "coord": {
                0: "voxels_num",
            },
            "grid_coord": {
                0: "voxels_num",
            },
            "feat": {
                0: "voxels_num",
            },
            "serialized_code": {
                1: "voxels_num",
            },
            "serialized_order": {
                1: "voxels_num",
            },
            "serialized_inverse": {
                1: "voxels_num",
            },
        }

        torch.onnx.export(
            model,
            input_dict,
            output_path,
            export_params=export_params,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            verbose=True,
            do_constant_folding=False)

    print("Exported to ONNX format successfully.")

    print("Attempting to fix the graph (TopK's K becoming a tensor)")

    import onnx_graphsurgeon as gs

    model = onnx.load(output_path)
    graph = gs.import_onnx(model)

    # Fix TopK
    """ topk_nodes = [node for node in graph.nodes if node.op == "TopK"]
    assert len(topk_nodes) == 1
    topk = topk_nodes[0]
    k = model_cfg.num_proposals
    topk.inputs[1] = gs.Constant("K", values=np.array([k], dtype=np.int64))
    topk.outputs[0].shape = [1, k]
    topk.outputs[0].dtype = topk.inputs[0].dtype if topk.inputs[0].dtype else np.float32
    topk.outputs[1].shape = [1, k]
    topk.outputs[1].dtype = np.int64 """

    graph.cleanup().toposort()
    output_path = output_path.replace(".onnx", "_fixed.onnx")
    onnx.save_model(gs.export_onnx(graph), output_path)

    print(f"(Fixed) ONNX exported to {output_path}", flush=True)


if __name__ == "__main__":
    main()
