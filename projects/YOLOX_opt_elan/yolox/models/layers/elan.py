from typing import Optional

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from .csp_layer_opt import DarknetBottleneckOpt


class ELAN(BaseModule):
    """Efficient Layer Aggregation Network, a.k.a ELAN.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): Size of kernel. Default: 1.
        expand_ratio (int): The kernel size of the convolution. Default: 1.0.
        num_blocks (int): Number of blocks. Default: 3.
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expand_ratio: float = 1.0,
        num_blocks: int = 3,
        add_identity: bool = True,
        use_depthwise: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="ReLU"),
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        mid_channels: int = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.final_conv = ConvModule(
            (num_blocks + 1) * mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.blocks = nn.Sequential(
            *[
                DarknetBottleneckOpt(
                    mid_channels,
                    mid_channels,
                    kernel_size,
                    1.0,
                    add_identity,
                    use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        x_short = self.short_conv(x)
        el = [x_short]
        x_main = self.main_conv(x)
        for m in self.blocks:
            x_main = m(x_main)
            el.append(x_main)
        x_final = torch.cat(el, dim=1)
        return self.final_conv(x_final)
