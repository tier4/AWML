# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.models.layers import make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.middle_encoders import SparseEncoder
from mmdet3d.registry import MODELS

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor

import numpy as np
import torch


@MODELS.register_module()
class BEVFusionSparseEncoder(SparseEncoder):
    r"""Sparse encoder for BEVFusion. The difference between this
    implementation and that of ``SparseEncoder`` is that the shape order of 3D
    conv is (H, W, D) in ``BEVFusionSparseEncoder`` rather than (D, H, W) in
    ``SparseEncoder``. This difference comes from the implementation of
    ``voxelization``.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
    """

    def __init__(
        self,
        in_channels,
        aug_features,
        aug_features_min_values,
        aug_features_max_values,
        num_aug_features,
        sparse_shape,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
        return_middle_feats=False,
    ):
        super(SparseEncoder, self).__init__()
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.aug_features = aug_features
        self.aug_features_min_values = torch.tensor(aug_features_min_values).cuda()
        self.aug_features_max_values = torch.tensor(aug_features_max_values).cuda()
        self.num_aug_features = num_aug_features
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        self.return_middle_feats = return_middle_feats
        # Spconv init all weight on its own

        if aug_features:
            self.in_channels = in_channels * num_aug_features * 2
            self.exponents = 2 ** torch.arange(0, num_aug_features).to(torch.device("cuda")).float()

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        if self.order[0] != "conv":  # pre activate
            self.conv_input = make_sparse_convmodule(
                self.in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
                order=("conv",),
            )
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                self.in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
            )

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule, norm_cfg, self.base_channels, block_type=block_type
        )

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="spconv_down2",
            conv_type="SparseConv3d",
        )

    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features
                include:

            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_middle_feats is True, the
                module returns middle features.
        """

        if self.aug_features:
            num_points = voxel_features.shape[0]
            x = (voxel_features - self.aug_features_min_values.view(1, -1)) / (
                self.aug_features_max_values - self.aug_features_min_values
            ).view(1, -1)
            y = x.reshape(-1, 1) * np.pi * self.exponents.reshape(1, -1)
            y = y.reshape(num_points, -1)
            voxel_features = torch.cat([torch.cos(y), torch.sin(y)], dim=1)

        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)

        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features
