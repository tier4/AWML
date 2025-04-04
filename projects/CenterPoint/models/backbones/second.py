from typing import List, Optional, Sequence

from mmdet3d.models.backbones.second import SECOND as _SECOND
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig
from mmengine.logging import print_log


@MODELS.register_module(force=True)
class SECOND(_SECOND):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet."""

    def __init__(
        self,
        frozen_stages: Optional[List[int]] = None,
        in_channels: int = 128,
        out_channels: Sequence[int] = [128, 128, 256],
        layer_nums: Sequence[int] = [3, 5, 5],
        layer_strides: Sequence[int] = [2, 2, 2],
        norm_cfg: ConfigType = dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg: ConfigType = dict(type="Conv2d", bias=False),
        init_cfg: OptMultiConfig = None,
        pretrained: Optional[str] = None,
    ) -> None:
        super(SECOND, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self._frozen_stages = frozen_stages
        print_log(f"Frozen SECOND up to stage {frozen_stages}")
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters in every layer/stage."""
        if self._frozen_stages is None:
            return

        for i in self._frozen_stages:
            for params in self.blocks[i].parameters():
                params.requires_grad = False
            print_log(f"Freeze SECOND stage {i}.")
