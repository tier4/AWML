import math
from typing import Optional

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
#from mmdet.models import BACKBONES
#from mmdet.models.builder import BACKBONES 
from mmdet.registry import MODELS
from torch.nn.modules.batchnorm import _BatchNorm

from ..layers import ASPP, CSPLayerOpt


class Stem(nn.Module):
    """Stem module that is took away slice operations.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 3
        stride (int): The stride of the convolution. Default: 2
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="ReLU"),
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            *(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )
        )

    def forward(self, x):
        return self.conv(x)


@MODELS.register_module()
class CSPDarknetOpt(BaseModule):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX optimized for GPU acceleration.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 2.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_overwrite(list): Overwrite default arch settings. Default: None.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from awml_det2d.models import CSPDarknet
        >>> import torch
        >>> self = CSPDarknetOpt()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 960, 960)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 128, 120, 120)
        (1, 256, 60, 60)
        (1, 512, 30, 30)
    """

    # From left to right:
    # in_channels, out_channels, num_blocks, expand_ratio, add_identity, use_aspp
    arch_settings = {
        "P5": [
            [32, 64, 1, 0.5, True, False],
            [64, 128, 3, 0.5, True, False],
            [128, 256, 3, 0.5, True, False],
            [256, 512, 1, 0.5, False, False],
        ],
        "P5_ASPP": [
            [32, 64, 1, 0.5, False, True],
            [64, 128, 3, 0.5, False, True],
            [128, 256, 3, 0.5, False, True],
            [256, 512, 1, 0.5, True, False],
        ],
    }

    def __init__(
        self,
        arch="P5",
        deepen_factor=2.0,
        widen_factor=1.0,
        out_indices=(2, 3, 4),
        frozen_stages=-1,
        use_depthwise=False,
        aspp_dilations=(1, 6, 12, 18),
        arch_overwrite=None,
        conv_cfg=None,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="ReLU"),
        norm_eval=False,
        init_cfg=dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ) -> None:
        super().__init__(init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_overwrite:
            arch_setting = arch_overwrite
        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError(
                "frozen_stages must be in range(-1, " "len(arch_setting) + 1). But received " f"{frozen_stages}"
            )

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.stem = Stem(
            3,
            int(arch_setting[0][0] * widen_factor),
            kernel_size=3,
            stride=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.layers = ["stem"]

        for i, (in_channels, out_channels, num_blocks, expand_ratio, add_identity, use_aspp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            stage.append(conv_layer)
            if use_aspp:
                aspp = ASPP(
                    out_channels,
                    out_channels,
                    dilations=aspp_dilations,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                stage.append(aspp)
            csp_layer = CSPLayerOpt(
                out_channels,
                out_channels,
                kernel_size=3,
                expand_ratio=expand_ratio,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            stage.append(csp_layer)
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknetOpt, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
