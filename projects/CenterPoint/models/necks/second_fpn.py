from typing import List, Optional

from mmdet3d.models.necks import SECONDFPN as _SECONDFPN
from mmdet3d.registry import MODELS
from mmengine.logging import print_log


@MODELS.register_module(force=True)
class SECONDFPN(_SECONDFPN):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to
            [dict(type='Kaiming', layer='ConvTranspose2d'),
             dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)].
    """

    def __init__(
        self,
        frozen_stages: Optional[List[int]] = None,
        in_channels=[128, 128, 256],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        conv_cfg=dict(type="Conv2d", bias=False),
        use_conv_for_no_stride=False,
        init_cfg=[
            dict(type="Kaiming", layer="ConvTranspose2d"),
            dict(type="Constant", layer="NaiveSyncBatchNorm2d", val=1.0),
        ],
    ):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            upsample_strides=upsample_strides,
            norm_cfg=norm_cfg,
            upsample_cfg=upsample_cfg,
            conv_cfg=conv_cfg,
            use_conv_for_no_stride=use_conv_for_no_stride,
            init_cfg=init_cfg,
        )
        self._frozen_stages = frozen_stages

        print_log(f"Frozen SECONDFPN up to stage {frozen_stages}")
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters in every layer/stage."""
        if self._frozen_stages is None:
            return

        for i in self._frozen_stages:
            for params in self.deblocks[i].parameters():
                params.requires_grad = False
            print_log(f"Freeze SECONDFPN stage {i}.")
