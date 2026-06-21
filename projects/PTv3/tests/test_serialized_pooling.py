import unittest

import torch
from models.point_transformer_v3.point_transformer_v3m1_base import SerializedPooling, build_serialized_pooling_meta
from models.utils.structure import Point


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
        common = dict(stride=2, reduce="max", shuffle_orders=False, traceable=True)
        train_module = SerializedPooling(6, 8, export_mode=False, **common)
        export_module = SerializedPooling(6, 8, export_mode=True, export_stage_index=0, **common)
        for module in (train_module, export_module):
            module.norm = None
            module.act = None
        export_module.load_state_dict(train_module.state_dict())

        train_out = train_module(self._make_point())
        export_point = self._make_point()
        meta, _ = build_serialized_pooling_meta(
            export_point.grid_coord, export_point.serialized_code, export_point.serialized_order, stride=2
        )
        export_point["serialized_pooling"] = [meta]
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
            left, right = train_out[key], export_out[key]
            if left.dtype.is_floating_point:
                torch.testing.assert_close(left, right, msg=f"Mismatch for {key}")
            else:
                self.assertTrue(torch.equal(left, right), f"Mismatch for {key}")


if __name__ == "__main__":
    unittest.main()
