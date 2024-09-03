from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType


@MODELS.register_module()
class FrustumFeatureEncoder(nn.Module):
    """Frustum Feature Encoder.

    Args:
        in_channels (int): Number of input features, either x, y, z or
            x, y, z, r. Defaults to 4.
        feat_channels (Sequence[int]): Number of features in each of the N
            FFELayers. Defaults to [].
        with_distance (bool): Whether to include Euclidean distance to points.
            Defaults to False.
        with_cluster_center (bool): Whether to include cluster center.
            Defaults to False.
        norm_cfg (dict or :obj:`ConfigDict`): Config dict of normalization
            layers. Defaults to dict(type='BN1d', eps=1e-5, momentum=0.1).
        with_pre_norm (bool): Whether to use the norm layer before input ffe
            layer. Defaults to False.
        feat_compression (int, optional): The frustum feature compression
            channels. Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: Sequence[int] = [],
                 with_distance: bool = False,
                 with_cluster_center: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-5, momentum=0.1),
                 with_pre_norm: bool = False,
                 feat_compression: Optional[int] = None) -> None:
        super(FrustumFeatureEncoder, self).__init__()
        assert len(feat_channels) > 0

        if with_distance:
            in_channels += 1
        if with_cluster_center:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center

        feat_channels = [self.in_channels] + list(feat_channels)
        if with_pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.in_channels)[1]
        else:
            self.pre_norm = None

        ffe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            if i == len(feat_channels) - 2:
                ffe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                ffe_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters, bias=False),
                        norm_layer, nn.ReLU(inplace=True)))
        self.ffe_layers = nn.ModuleList(ffe_layers)
        self.compression_layers = None
        if feat_compression is not None:
            self.compression_layers = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_compression),
                nn.ReLU(inplace=True))

    def forward(self, voxel_dict: dict) -> dict:
        features = voxel_dict['voxels']
        coors = voxel_dict['coors']
        features_ls = [features]

        # Sort
        indices_2 = torch.argsort(coors[:, 2])
        sorted_coors = coors[indices_2]
        indices_1 = torch.argsort(sorted_coors[:, 1])
        sorted_coors = sorted_coors[indices_1]
        indices = indices_2[indices_1]
        # Calculate differences between rows
        min_value_row = torch.full((1, sorted_coors.size(1)), torch.iinfo(sorted_coors.dtype).min, dtype=sorted_coors.dtype, device=sorted_coors.device)
        sorted_coors_ex = torch.cat([min_value_row, sorted_coors], dim=0)
        diffs = sorted_coors_ex[1:] - sorted_coors_ex[:-1]
        # Create a mask for rows that are unique
        unique_mask = (diffs != 0).any(dim=1)
        # Extract unique rows
        voxel_coors = sorted_coors[unique_mask]
        # Calculate inverse indices
        inverse_map = torch.cumsum(unique_mask.to(torch.int64), dim=0) - 1
        inverse_map = inverse_map[torch.argsort(indices)]

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Find distance of x, y, and z from frustum center
        index = inverse_map.unsqueeze(-1).expand(-1, features.size(1))
        if self._with_cluster_center:
            voxel_sum = torch.zeros(inverse_map.max() + 1, features.size(1), device=features.device, dtype=features.dtype)
            voxel_count = voxel_sum.clone()
            ones_tensor = torch.ones_like(features)

            voxel_sum = voxel_sum.scatter_add_(dim=0, index=index, src=features)
            voxel_count = voxel_count.scatter_add_(dim=0, index=index, src=ones_tensor)
            voxel_mean = voxel_sum / voxel_count

            points_mean = voxel_mean[inverse_map]
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        if self.pre_norm is not None:
            features = self.pre_norm(features)

        point_feats = []
        for ffe in self.ffe_layers:
            features = ffe(features)
            point_feats.append(features)

        # TODO(amadeuszsz): Check metrics for both options
        # voxel_feats = torch.zeros(inverse_map.max() + 1, features.size(1), device=features.device, dtype=features.dtype)
        voxel_feats = torch.full((inverse_map.max() + 1, features.size(1)), torch.finfo(features.dtype).min, device=features.device, dtype=features.dtype)
        voxel_feats.scatter_reduce_(dim=0, index=inverse_map.unsqueeze(-1).expand(-1, features.size(1)), src=features, reduce='amax', include_self=True)

        if self.compression_layers is not None:
            voxel_feats = self.compression_layers(voxel_feats)

        voxel_dict['voxel_feats'] = voxel_feats
        voxel_dict['voxel_coors'] = voxel_coors
        voxel_dict['point_feats'] = point_feats

        return voxel_dict
