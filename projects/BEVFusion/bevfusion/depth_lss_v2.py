from typing import Tuple

import torch
from mmdet3d.registry import MODELS
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
            geom_feats, kept, ranks, indices = geom_feats_precomputed
            x, depth_softmax = self.get_cam_feats(img)
            x = self.bev_pool_precomputed(x, depth_softmax, geom_feats, kept, ranks, indices)

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

        return x

    def bev_pool_aux(self, geom_feats):
        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W
        assert C == 3

        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(0, Nprime - 1, dtype=torch.int, device=geom_feats.device)
        ranks_feat = torch.range(0, Nprime // D - 1, dtype=torch.int, device=geom_feats.device)
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
            return None, None, None, None

        geom_feats, ranks_depth, ranks_feat = geom_feats[kept], ranks_depth[kept], ranks_feat[kept]

        # nx is the total number of voxels/cells in the BEV grid
        # nx[0] is x, nx[1] is y, nx[2] is z
        ranks_bev = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        indices = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = ranks_bev[indices], ranks_depth[indices], ranks_feat[indices]

        intervals = self.compute_intervals(ranks_bev)
        if intervals is None:
            return None, None, None, None, None

        interval_starts, interval_lengths = intervals
        return (
            ranks_bev.int().contiguous(),
            ranks_depth.int().contiguous(),
            ranks_feat.int().contiguous(),
            interval_starts.int().contiguous(),
            interval_lengths.int().contiguous(),
        )

    def compute_intervals(self, ranks_bev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None

        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return interval_starts.int().contiguous(), interval_lengths.int().contiguous()

    def bev_pool(self, view_feats, depth_softmax, geom) -> torch.Tensor:
        """ """
        B, N, D, H, W, _ = geom.shape
        num_points = B * N * D * H * W

        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(0, num_points - 1, dtype=torch.int, device=geom.device)
        ranks_feat = torch.range(0, num_points // D - 1, dtype=torch.int, device=geom.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()

        B, N, C, fH, fW = view_feats.shape

        bev_feat = bev_pool_v2(
            depth_softmax,
            x,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            bev_feat_shape,
            interval_starts,
            interval_lengths,
            is_training,
        )
        return bev_feat


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
        self.downsample = DownSampleNet(downsample, self.out_channels, self.out_channels)

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)
        x = self.depthnet(x)

        depth_softmax = x[:, : self.D].softmax(dim=1)
        depth_softmax = depth_softmax.view(B, N, self.D, fH, fW)
        view_feats = x[:, self.D : (self.D + self.C)]
        view_feats = view_feats.view(B, N, self.C, fH, fW)
        return view_feats, depth_softmax
