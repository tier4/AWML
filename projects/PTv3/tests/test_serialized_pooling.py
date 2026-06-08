import sys
import types
import unittest
from pathlib import Path

import torch

if not hasattr(torch, "inference_mode"):
    torch.inference_mode = torch.no_grad

try:
    import torch_scatter  # noqa: F401
except ModuleNotFoundError:
    torch_scatter = types.ModuleType("torch_scatter")

    def _segment_csr(src, indptr, reduce="sum"):
        outputs = []
        for start, end in zip(indptr[:-1].tolist(), indptr[1:].tolist()):
            segment = src[start:end]
            if reduce == "sum":
                outputs.append(segment.sum(dim=0))
            elif reduce == "mean":
                outputs.append(segment.mean(dim=0))
            elif reduce == "max":
                outputs.append(segment.max(dim=0).values)
            elif reduce == "min":
                outputs.append(segment.min(dim=0).values)
            else:
                raise NotImplementedError(f"Unsupported reduce mode: {reduce}")
        return torch.stack(outputs, dim=0)

    torch_scatter.segment_csr = _segment_csr
    sys.modules["torch_scatter"] = torch_scatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.point_transformer_v3.point_transformer_v3m1_base import SerializedPooling, SerializedPoolingMeta
from models.scatter.functional import argsort
from models.utils.structure import Point


def _pooling_depth(stride):
    depth = 0
    value = int(stride)
    while value > 1:
        depth += 1
        value >>= 1
    return depth


def _make_serialized_pooling_metadata(point, stride):
    depth = _pooling_depth(stride)
    pooled_code = point.serialized_code >> (depth * 3)
    indices = point.serialized_order[0]
    sorted_code = pooled_code[0].index_select(0, indices)
    run_start = torch.cat(
        [
            torch.ones_like(sorted_code[:1], dtype=torch.bool),
            sorted_code[1:] != sorted_code[:-1],
        ],
        dim=0,
    )
    starts = torch.nonzero(run_start, as_tuple=False).flatten()
    indptr = torch.cat([starts, torch.tensor([indices.numel()], dtype=starts.dtype, device=starts.device)], dim=0)
    cluster_sorted = torch.cumsum(run_start.to(dtype=indices.dtype), dim=0) - 1
    cluster = torch.zeros_like(cluster_sorted)
    cluster.scatter_(0, indices, cluster_sorted)
    head_indices = indices.index_select(0, indptr[:-1])
    grid_coord = point.grid_coord.index_select(0, head_indices) >> depth
    serialized_code = pooled_code.index_select(1, head_indices)
    serialized_order = torch.stack([argsort(code) for code in serialized_code], dim=0)
    serialized_inverse = torch.zeros_like(serialized_order).scatter_(
        dim=1,
        index=serialized_order,
        src=torch.arange(0, serialized_code.shape[1], device=serialized_order.device).repeat(
            serialized_code.shape[0], 1
        ),
    )
    return SerializedPoolingMeta(
        indices=indices,
        indptr=indptr,
        cluster=cluster,
        head_indices=head_indices,
        grid_coord=grid_coord,
        serialized_order=serialized_order,
        serialized_inverse=serialized_inverse,
    )


class TestSerializedPooling(unittest.TestCase):
    def setUp(self):
        self.grid_coord = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 2, 1],
                [3, 2, 1],
                [2, 3, 1],
                [3, 3, 1],
                [0, 0, 2],
                [1, 0, 2],
            ],
            dtype=torch.int32,
        )
        self.batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.int64)
        self.feat = torch.randn(self.grid_coord.shape[0], 6)
        self.coord = self.grid_coord.to(torch.float32)
        self.sparse_shape = torch.tensor([16, 16, 16], dtype=torch.int64)
        self.depth = 6

    def _make_point(self):
        point = Point(
            coord=self.coord.clone(),
            grid_coord=self.grid_coord.clone(),
            feat=self.feat.clone(),
            batch=self.batch.clone(),
            sparse_shape=self.sparse_shape.clone(),
        )
        point.serialization(order=["z", "z-trans"], depth=self.depth, shuffle_orders=False)
        return point

    def test_export_mode_matches_train_time(self):
        torch.manual_seed(0)
        train_module = SerializedPooling(
            6,
            8,
            stride=2,
            reduce="max",
            shuffle_orders=False,
            traceable=True,
            export_mode=False,
        )
        export_module = SerializedPooling(
            6,
            8,
            stride=2,
            reduce="max",
            shuffle_orders=False,
            traceable=True,
            export_mode=True,
            export_stage_index=0,
        )
        train_module.norm = None
        train_module.act = None
        export_module.norm = None
        export_module.act = None
        export_module.load_state_dict(train_module.state_dict())

        train_out = train_module(self._make_point())
        export_point = self._make_point()
        export_point["serialized_pooling"] = [_make_serialized_pooling_metadata(export_point, stride=2)]
        export_out = export_module(export_point)

        tensor_keys = [
            "feat",
            "grid_coord",
            "serialized_order",
            "serialized_inverse",
            "batch",
            "sparse_shape",
            "pooling_inverse",
        ]

        for key in tensor_keys:
            left = train_out[key]
            right = export_out[key]
            if left.dtype.is_floating_point:
                if hasattr(torch.testing, "assert_close"):
                    torch.testing.assert_close(left, right, msg=f"Mismatch for {key}")
                else:
                    torch.testing.assert_allclose(left, right, msg=f"Mismatch for {key}")
            else:
                self.assertTrue(torch.equal(left, right), f"Mismatch for {key}")


if __name__ == "__main__":
    unittest.main()
