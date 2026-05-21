# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# EVAViT is loaded lazily: it depends on fvcore, which many containers omit.
# VoVNet / VoVNetCP do not need fvcore.
from __future__ import annotations

from typing import TYPE_CHECKING

from .vovnet import VoVNet
from .vovnetcp import VoVNetCP

if TYPE_CHECKING:
    from .eva_vit import EVAViT

__all__ = ["VoVNet", "VoVNetCP", "EVAViT"]


def __getattr__(name: str):
    if name == "EVAViT":
        from .eva_vit import EVAViT as _EVAViT

        return _EVAViT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
