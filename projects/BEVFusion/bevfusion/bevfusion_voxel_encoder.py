from typing import Optional, Tuple

import numpy as np
import torch
from mmdet3d.models.voxel_encoders.utils import PFNLayer, get_paddings_indicator
from mmdet3d.registry import MODELS
from torch import Tensor, nn


@MODELS.register_module()
class HardSimpleVoxelSinCosEncoder(nn.Module):
    def __init__(
        self, min_norm_values: Tuple[float], max_norm_values: Tuple[float], in_channels: Optional[int] = 4
    ) -> None:
        """
        Simple voxel encoder that only performs mean pooling on the normalize features, and then
        performs sin-cos (fourier encoding) on each voxel channels.

        The output shape of each voxel is (N, feature_channels*2).
        Args:
            min_norm_values (Tuple[float]): Minimum values for the features.
            max_norm_values (Tuple[float]): Maximum values for the features.
            in_channels (int): Number of input channels.
        """
        super().__init__()

        # Create PillarFeatureNet layers
        self.in_channels = in_channels

        # Convert the ((x - min) / (max - min)) * pi * exponents to x * scale + bias for folding them into one OP
        min_norm_values = torch.tensor(min_norm_values)
        max_norm_values = torch.tensor(max_norm_values)
        # Let alpha = pi * exponents, beta = max - min
        # y = ((x - min) / beta) * alpha
        # y = alpha / beta * (x - min)
        # y = (alpha / beta) * x - (alpha / beta) * min
        # Therefore, scale = alpha / beta, bias = - (alpha * min) / beta
        # y = scale * x + bias
        exponents = (2 ** torch.arange(0, self.in_channels)).float()
        alpha = (torch.pi * exponents).unsqueeze(0)  # (1, C)
        beta = (max_norm_values - min_norm_values).unsqueeze(1)  # (C, 1)
        scale = alpha / beta
        bias = -(alpha * min_norm_values.unsqueeze(1)) / beta  # (C, C)

        self.register_buffer("exponent_scale", scale.unsqueeze(0), persistent=False)  # (1, C, C)
        self.register_buffer("exponent_bias", bias.unsqueeze(0), persistent=False)  # (1, C, C)

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor, *args, **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C) in (x, y, z, intensity, time_lag) if C is 5, (x, y, z, time_lag) if C is 4.
            num_points (torch.Tensor): Number of points in each pillar in shape (M).
            coors (torch.Tensor): Coordinates of each voxel in (M, [4]), which is (batch_idx, z_idx, y_idx, x_idx).

        Returns:
            torch.Tensor: Features of pillars in shape (M, C*C*2).

        """
        # Mean in the voxel
        # (N, M, C) -> (N, C)
        voxel_mean_features = (
            features.sum(dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        ).contiguous()

        # x * scale + bias, (1, C, C) + (1, C, C) * (N, C, 1) -> (N, C, C)
        # FMA (fused multiply-add): y = bias + scale * voxel_mean_features
        y = torch.addcmul(self.exponent_bias, self.exponent_scale, voxel_mean_features.unsqueeze(-1))
        # SinCos encoding
        # (N*C, C) -> (N, C*C)
        y = y.reshape(-1, self.in_channels * self.in_channels)
        # (N, C*C) -> (N, C*C*2)
        voxel_fourier_features = torch.cat([torch.cos(y), torch.sin(y)], dim=1)

        return voxel_fourier_features
