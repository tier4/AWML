import math
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmdet3d.registry import MODELS
from mmengine.logging import print_log
from torch import nn

from .depth_lss import BaseViewTransform, DepthLSSNet, DownSampleNet, LidarDepthImageNet
from .ops import bev_pool_v2


class BaseViewTransformV2(BaseViewTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        collapse_z: bool = True,
        expand_batch_axis: bool = False,
        visualize_bev_feat: bool = False,
    ):
        """
        Args:
            collapse_z: collapse the Z axis of the BEV grid
            expand_batch_axis: expand the batch axis of the inputs to bev pool if this is set to True.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            visualize_bev_feat=visualize_bev_feat,
        )
        self.collapse_z = collapse_z
        self.expand_batch_axis = expand_batch_axis

    def get_cam_feats(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        img,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        camera_intrinsics_inverse,
        img_aug_matrix_inverse,
        lidar_aug_matrix_inverse,
        geom_feats_precomputed,
    ):
        if geom_feats_precomputed is not None:
            ranks_bev, ranks_depth, ranks_feat = geom_feats_precomputed
            x, depth_softmax = self.get_cam_feats(img)
            x = self.bev_pool_precomputed(x, depth_softmax, ranks_bev, ranks_depth, ranks_feat)
            
            # No return depth predictions when precomputed geometry features are used
            depth_softmax = None
        
        else:
            intrins = camera_intrinsics[..., :3, :3]
            post_rots = img_aug_matrix[..., :3, :3]
            post_trans = img_aug_matrix[..., :3, 3]
            camera2lidar_rots = camera2lidar[..., :3, :3]
            camera2lidar_trans = camera2lidar[..., :3, 3]

            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]

            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                torch.inverse(intrins),
                torch.inverse(post_rots),
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )

            # depth is not connected to the calibration
            # on_img is
            # is also flattened_indices
            (
                view_feats,
                depth_softmax,
            ) = self.get_cam_feats(img)
            x = self.bev_pool(view_feats, depth_softmax, geom)
         
        return x, depth_softmax

    def bev_pool_aux(self, geom_feats):
        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W
        assert C == 3

        # record the index of selected points for acceleration purpose
        ranks_depth = torch.arange(0, Nprime, dtype=torch.int, device=geom_feats.device)
        ranks_feat = torch.arange(0, Nprime // D, dtype=torch.int, device=geom_feats.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=geom_feats.device, dtype=torch.long) for ix in range(B)]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )

        if len(kept) == 0:
            return None, None, None

        geom_feats, ranks_depth, ranks_feat = geom_feats[kept], ranks_depth[kept], ranks_feat[kept]
        
        # Switch x and y to match the order of the BEV grid
        ranks_bev = (
            geom_feats[:, 3] * (self.nx[2] * self.nx[1] * self.nx[0])
            + geom_feats[:, 2] * (self.nx[1] * self.nx[0])
            + geom_feats[:, 0] * self.nx[1]
            + geom_feats[:, 1]
        )
        indices = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = ranks_bev[indices], ranks_depth[indices], ranks_feat[indices]
        return (
            ranks_bev.int().contiguous(),
            ranks_depth.int().contiguous(),
            ranks_feat.int().contiguous(),
        )

    def compute_intervals(self, ranks_bev: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if ranks_bev is None:
            return None, None

        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None

        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return interval_starts.int().contiguous(), interval_lengths.int().contiguous()

    def bev_pool(self, view_feats, depth_softmax, geom) -> torch.Tensor:
        """ """
        ranks_bev, ranks_depth, ranks_feat = self.bev_pool_aux(geom)
        interval_starts, interval_lengths = self.compute_intervals(ranks_bev)
        bev_feat = self.compute_bev_pool(
            view_feats, depth_softmax, ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths
        )
        return bev_feat

    def compute_bev_pool(
        self, view_feats, depth_softmax, ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths
    ):
        """Compute the BEV pool for the given view features, depth softmax, ranks, and intervals."""
        if interval_starts is None:
            print_log("warning ---> no points within the predefined bev receptive field")
            dummy = torch.zeros(
                size=[view_feats.shape[0], view_feats.shape[2], self.nx[2], self.nx[1], self.nx[0]],
                dtype=view_feats.dtype,
                device=view_feats.device,
            )
            if self.collapse_z:
                dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy

        if self.expand_batch_axis:
            view_feats = view_feats.unsqueeze(0)
            depth_softmax = depth_softmax.unsqueeze(0)

        # permute view_feats from (B, N, C, fH, fW) to (B, N, fH, fW, C)
        view_feats = view_feats.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (
            depth_softmax.shape[0],
            int(self.nx[2]),
            int(self.nx[1]),
            int(self.nx[0]),
            view_feats.shape[-1],
        )  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(
            depth=depth_softmax,
            feat=view_feats,
            ranks_depth=ranks_depth,
            ranks_feat=ranks_feat,
            ranks_bev=ranks_bev,
            interval_starts=interval_starts,
            interval_lengths=interval_lengths,
            bev_feat_shape=bev_feat_shape,
            is_training=self.training,
        )

        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)

        if self.visualize_bev_feat:
            self.plot_bev_feat(bev_feat)

        return bev_feat
     
    def bev_pool_precomputed(self, view_feats, depth_softmax, ranks_bev, ranks_depth, ranks_feat):
        interval_starts, interval_lengths = self.compute_intervals(ranks_bev)
        bev_feat = self.compute_bev_pool(
            view_feats, depth_softmax, ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths
        )
        return bev_feat 


@MODELS.register_module()
class LSSTransformV2(BaseViewTransformV2):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(self.in_channels, self.D + self.C, 1)
        self.downsample = DownSampleNet(downsample, out_channels, out_channels)

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)
        x = self.depthnet(x)

        depth_softmax = x[:, : self.D].softmax(dim=1)
        depth_softmax = depth_softmax.view(B, N, self.D, fH, fW)
        view_feats = x[:, self.D : (self.D + self.C)]
        view_feats = view_feats.view(B, N, self.C, fH, fW)
        return view_feats, depth_softmax
    
    def forward(self, *args, **kwargs):
        x, depth_softmax = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x, depth_softmax
    