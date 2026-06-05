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


@MODELS.register_module()
class BEVFusionVoxelFeatureNet(HardSimpleVoxelSinCosEncoder):
    """BEVFusion Voxel Encoder Feature Net.

    The network is same as HardSimpleVoxelSinCosEncoder, but it performs PFNLayers on the
    offset features, for example, distances. After that, it concatenates the fourier features and the PFN features
    along the channel dimension for each voxel.

    Args:
        min_norm_values (Tuple[float]): Minimum values for the features.
        max_norm_values (Tuple[float]): Maximum values for the features.
        in_channels (int): Number of input channels.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
    """

    def __init__(
        self,
        min_norm_values: Optional[Tuple[float]] = None,
        max_norm_values: Optional[Tuple[float]] = None,
        in_channels: Optional[int] = 4,
        feat_channels: Optional[tuple] = (64,),
        with_distance: Optional[bool] = False,
        with_cluster_center: Optional[bool] = True,
        with_voxel_center: Optional[bool] = True,
        voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 4),
        point_cloud_range: Optional[Tuple[float]] = (0, -40, -3, 70.4, 40, 1),
        norm_cfg: Optional[dict] = dict(type="BN1d", eps=1e-3, momentum=0.01),
        mode: Optional[str] = "max",
        legacy: Optional[bool] = True,
    ):

        super(BEVFusionVoxelFeatureNet, self).__init__(
            min_norm_values=min_norm_values, max_norm_values=max_norm_values, in_channels=in_channels
        )
        assert len(feat_channels) > 0
        self.legacy = legacy
        pfn_in_channels = in_channels
        if with_cluster_center:
            pfn_in_channels += 3
        if with_voxel_center:
            pfn_in_channels += 3
        if with_distance:
            pfn_in_channels += 1

        assert pfn_in_channels > 0, "pfn_in_channels must be greater than 0"
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center

        # Create VoxelFeatureNet layers
        feat_channels = [pfn_in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer, mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor, *args, **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C) in (x, y, z, intensity, time_lag) if C is 5, (x, y, z, time_lag) if C is 4.
            num_points (torch.Tensor): Number of points in each pillar in shape (M).
            coors (torch.Tensor): Coordinates of each voxel in (M, [4]), which is (batch_idx, z_idx, y_idx, x_idx).

        Returns:
            torch.Tensor: Features of pillars in shape (M, C*C*2 + feat_channels[-1]).
        """
        # (M, C*C*2)
        voxel_fourier_features = super().forward(features, num_points, coors)

        # Normalize the features
        norm_features = (features - self.min_norm_values.view(1, -1)) / (
            (self.max_norm_values - self.min_norm_values).view(1, -1)
        )

        # Offset features
        max_points_per_voxel = features.shape[1]

        features_ls = [norm_features]
        # Find distance of x, y, and z from cluster center, mapped to [-1,   1] if available
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            # Map to [0, 1] if available
            # if self.min_norm_values is not None and self.max_norm_values is not None:
            #     voxel_size = features.new_tensor([self.vx, self.vy, self.vz])
            #     f_cluster = f_cluster / voxel_size
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :3])
                f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
                f_center[:, :, 2] = features[:, :, 2] - (coors[:, 1].to(dtype).unsqueeze(1) * self.vz + self.z_offset)
            else:
                f_center = features[:, :, :3]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx + self.x_offset
                )
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy + self.y_offset
                )
                f_center[:, :, 2] = f_center[:, :, 2] - (
                    coors[:, 1].type_as(features).unsqueeze(1) * self.vz + self.z_offset
                )

            # if self.min_norm_values is not None and self.max_norm_values is not None:
            #     f_center = f_center / (voxel_size * 0.5)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        voxel_feature_offsets = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        mask = get_paddings_indicator(num_points, max_points_per_voxel, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_feature_offsets)
        voxel_feature_offsets *= mask

        # PFN
        for pfn in self.pfn_layers:
            voxel_feature_offsets = pfn(voxel_feature_offsets, num_points)

        # Concat
        features = torch.cat([voxel_fourier_features, voxel_feature_offsets.squeeze(1)], dim=-1)

        return features


# @MODELS.register_module()
# class BEVFusionVoxelSinCosEncoder(nn.Module):
#     def __init__(self,
#                  min_norm_values: Tuple[float],
#                  max_norm_values: Tuple[float],
#                  time_lag_channel_index: int = 3,
#                  time_exp_factor: Optional[float] = None,
#                  feat_channels: Optional[tuple] = (16, ),
#                  in_channels: Optional[int] = 4,
#                  with_distance: Optional[bool] = False,
#                  with_cluster_center: Optional[bool] = True,
#                  with_voxel_center: Optional[bool] = True,
#                  voxel_size: Optional[Tuple[float]] = (0.2, 0.2, 4),
#                  point_cloud_range: Optional[Tuple[float]] = (0, -40, -3, 70.4,
#                                                               40, 1),
#                  norm_cfg: Optional[dict] = dict(
#                      type='BN1d', eps=1e-3, momentum=0.01),
#                  mode: Optional[str] = 'max'):
#         super(BEVFusionVoxelSinCosEncoder, self).__init__()

#         self._with_distance = with_distance
#         self._with_cluster_center = with_cluster_center
#         self._with_voxel_center = with_voxel_center
#         # Create PillarFeatureNet layers
#         self.in_channels = in_channels

#         # Need pillar (voxel) size and x/y offset in order to calculate offset
#         self.vx = voxel_size[0]
#         self.vy = voxel_size[1]
#         self.vz = voxel_size[2]
#         self.x_offset = self.vx / 2 + point_cloud_range[0]
#         self.y_offset = self.vy / 2 + point_cloud_range[1]
#         self.z_offset = self.vz / 2 + point_cloud_range[2]
#         self.point_cloud_range = point_cloud_range

#         self.xyz_channels = 3
#         feat_offset_channels = in_channels - self.xyz_channels
#         if with_cluster_center:
#             feat_offset_channels += 3
#         if with_voxel_center:
#             feat_offset_channels += 3
#         if with_distance:
#             feat_offset_channels += 1

#         feat_channels = [feat_offset_channels] + list(feat_channels)
#         assert len(feat_channels) > 0, "feat_channels must be greater than 0"
#         pfn_layers = []
#         for i in range(len(feat_channels) - 1):
#             in_filters = feat_channels[i]
#             out_filters = feat_channels[i + 1]
#             if i < len(feat_channels) - 2:
#                 last_layer = False
#             else:
#                 last_layer = True
#             pfn_layers.append(
#                 PFNLayer(
#                     in_filters,
#                     out_filters,
#                     norm_cfg=norm_cfg,
#                     last_layer=last_layer,
#                     mode=mode))
#         self.pfn_layers = nn.ModuleList(pfn_layers)

#         self.time_lag_channel_index = time_lag_channel_index
#         self.time_exp_factor = time_exp_factor

#         self.register_buffer("min_norm_values", torch.tensor(min_norm_values))
#         self.register_buffer("max_norm_values", torch.tensor(max_norm_values))
#         self.register_buffer("voxel_size", torch.tensor([self.vx, self.vy, self.vz]))
#         self.register_buffer("exponents", (2 ** torch.arange(0, self.xyz_channels)).float())

#     def forward(self, features: Tensor, num_points: Tensor, coors: Tensor,
#                 *args, **kwargs) -> Tensor:
#         """Forward function.

#         Args:
#             features (torch.Tensor): Point features or raw points in shape
#                 (N, M, C).
#             num_points (torch.Tensor): Number of points in each pillar in shape (M).
#             coors (torch.Tensor): Coordinates of each voxel in (M, [4]), which is (batch_idx, z_idx, y_idx, x_idx).

#         Returns:
#             torch.Tensor: Features of pillars in shape (M, C).
#         """
#         num_voxels, max_points_per_voxel = features.shape[0], features.shape[1]

#         # Mean in the voxel
#         # (N, M, 3) -> (N, 3)
#         voxel_features = (features[:, :, :self.xyz_channels].sum(dim=1, keepdim=False) / num_points.type_as(features).view(
#                     -1, 1)).contiguous()

#         # min-max normalization, (N, 3) -> (N, 3)
#         voxel_features_norm = (voxel_features - \
#          self.min_norm_values[:self.xyz_channels].view(1, -1)) / ((self.max_norm_values[:self.xyz_channels] - self.min_norm_values[:self.xyz_channels]).view(1, -1))

#         # SinCos encoding
#         # (N, 3) -> (N*3, 1) * (1, ) * (1, 3) -> (N*3, 3)
#         y = voxel_features_norm.reshape(-1, 1) * np.pi * self.exponents.reshape(1, -1)
#         # (N*3, 3) -> (N, 3*3)
#         y = y.reshape(num_voxels, -1)
#         # (N, 3*3) -> (N, 3*3*2)
#         voxel_fourier_features = torch.cat([torch.cos(y), torch.sin(y)], dim=1)

#         # PFN
#         # Other features, for example, intensity or time_lag
#         other_features = features[:, :, self.xyz_channels:]

#         # Normalization
#         other_features_norm = (other_features - self.min_norm_values[self.xyz_channels:]) / (self.max_norm_values[self.xyz_channels:] - self.min_norm_values[self.xyz_channels:])

#         time_lag_feature_index = self.time_lag_channel_index - self.xyz_channels
#         # exponentiate time_lag features, it's higher when the normlized time lag is lower
#         # (1.0 when time_lag_features is 0.0)
#         if self.time_exp_factor is not None:
#             other_features_norm[:, :, time_lag_feature_index] = torch.exp(- other_features_norm[:, :, time_lag_feature_index] * self.time_exp_factor)
#         else:
#             # Inverse the time_lag feature
#             other_features_norm[:, :, time_lag_feature_index] = 1.0 - other_features_norm[:, :, time_lag_feature_index]

#         # Offsets
#         voxel_feature_offsets = [other_features_norm]
#         # Find distance of x, y, and z from cluster center
#         if self._with_cluster_center:
#             points_mean = features[:, :, :3].sum(
#                 dim=1, keepdim=True) / num_points.type_as(features).view(
#                     -1, 1, 1)

#             # f_cluster = (features[:, :, :3] - points_mean)
#             f_cluster = features[:, :, :3] - points_mean
#             voxel_feature_offsets.append(f_cluster)

#         # Find distance of x, y, and z from pillar center
#         dtype = features.dtype
#         if self._with_voxel_center:
#             f_center = torch.zeros_like(features[:, :, :3])
#             f_center[:, :, 0] = features[:, :, 0] - (
#                 coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
#                 self.x_offset)
#             f_center[:, :, 1] = features[:, :, 1] - (
#                 coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
#                 self.y_offset)
#             f_center[:, :, 2] = features[:, :, 2] - (
#                 coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
#                 self.z_offset)

#             # Map to [-1, 1]
#             # f_center = f_center / (self.voxel_size * 0.5)
#             voxel_feature_offsets.append(f_center)

#         if self._with_distance:
#             points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
#             voxel_feature_offsets.append(points_dist)

#         voxel_feature_offsets = torch.cat(voxel_feature_offsets, dim=-1)
#         # The feature decorations were calculated without regard to whether
#         # pillar was empty. Need to ensure that
#         # empty pillars remain set to zeros.
#         mask = get_paddings_indicator(num_points, max_points_per_voxel, axis=0)
#         mask = torch.unsqueeze(mask, -1).type_as(voxel_feature_offsets)
#         voxel_feature_offsets *= mask

#         # PFN
#         for pfn in self.pfn_layers:
#             voxel_feature_offsets = pfn(voxel_feature_offsets, num_points)

#         # Concat
#         features = torch.cat([voxel_fourier_features, voxel_feature_offsets.squeeze(1)], dim=-1)
#         return features
