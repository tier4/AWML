from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule


class ASPP(BaseModule):
    """Atrous Spatial Pyramid Pooling (ASPP) layer.

    Args:
        in_channels (int): The input channels of the ASPP layer.
        out_channels (int): The output channels of the ASPP layer.
        dilations (Sequence[int]): The dilations of the ASPP layer. Default: (1, 6, 12, 18).
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: List[int] = (1, 6, 12, 18),
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="ReLU"),
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg)

        self.aspp_modules = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1 if dilation == 1 else 3,
                    padding=0 if dilation == 1 else dilation,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                for dilation in dilations
            ]
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        hidden_channels: int = (len(dilations) + 1) * out_channels

        self.final_conv = ConvModule(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        aspp_outs = [aspp(x) for aspp in self.aspp_modules]
        x_pool = F.interpolate(
            self.global_avg_pool(x),
            size=x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        aspp_outs.append(x_pool)

        x = torch.cat(aspp_outs, dim=1)

        return self.final_conv(x)
