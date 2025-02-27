# Copyright (c) OpenMMLab. All rights reserved.
# Partially copied from https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/models/layers/spconv/overwrite_spconv/write_spconv2.py
# NOTE(knzo25): needed to overwrite our custom deployment oriented operations

from mmdet3d.models.layers.spconv.overwrite_spconv.write_spconv2 import _load_from_state_dict
from mmengine.registry import MODELS


def register_spconv2() -> bool:
    """This func registers spconv2.0 spconv ops to overwrite the default mmcv
    spconv ops."""
    try:
        from spconv.pytorch import (
            SparseConv2d,
            SparseConv4d,
            SparseConvTranspose2d,
            SparseConvTranspose3d,
            SparseInverseConv2d,
            SparseInverseConv3d,
            SparseModule,
            SubMConv2d,
            SubMConv4d,
        )

        from .sparse_conv import SparseConv3d, SubMConv3d
    except ImportError:
        return False
    else:
        MODELS._register_module(SparseConv2d, "SparseConv2d", force=True)
        MODELS._register_module(SparseConv3d, "SparseConv3d", force=True)
        MODELS._register_module(SparseConv4d, "SparseConv4d", force=True)

        MODELS._register_module(SparseConvTranspose2d, "SparseConvTranspose2d", force=True)
        MODELS._register_module(SparseConvTranspose3d, "SparseConvTranspose3d", force=True)

        MODELS._register_module(SparseInverseConv2d, "SparseInverseConv2d", force=True)
        MODELS._register_module(SparseInverseConv3d, "SparseInverseConv3d", force=True)

        MODELS._register_module(SubMConv2d, "SubMConv2d", force=True)
        MODELS._register_module(SubMConv3d, "SubMConv3d", force=True)
        MODELS._register_module(SubMConv4d, "SubMConv4d", force=True)
        SparseModule._version = 2
        SparseModule._load_from_state_dict = _load_from_state_dict
        return True
