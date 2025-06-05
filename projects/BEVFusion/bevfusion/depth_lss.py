# modify from https://github.com/mit-han-lab/bevfusion
from typing import Tuple

import torch
from mmdet3d.registry import MODELS
from torch import nn

from .ops import bev_pool


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class BaseViewTransform(nn.Module):

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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape

        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins_inverse,
        post_rots_inverse,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = post_rots_inverse.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(intrins_inverse)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool_aux(self, geom_feats):

        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W
        assert C == 3

        """ frustrum_numpy = geom_feats.cpu().numpy() """

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

        geom_feats = geom_feats[kept]

        """ data = {}
        data["frustum"] = frustrum_numpy
        data["kept"] = kept.cpu().numpy()
        import pickle
        with open("frustum.pkl", "wb") as f:
            pickle.dump(data, f) """

        # TODO(knzo25): make this more elegant
        D, H, W = self.nx[2], self.nx[0], self.nx[1]

        ranks = geom_feats[:, 0] * (W * D * B) + geom_feats[:, 1] * (D * B) + geom_feats[:, 2] * B + geom_feats[:, 3]
        indices = ranks.argsort()

        ranks = ranks[indices]
        geom_feats = geom_feats[indices]

        return geom_feats, kept, ranks, indices

    def bev_pool(self, x, geom_feats):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # Taken out of bev_pool for pre-computation
        geom_feats, kept, ranks, indices = self.bev_pool_aux(geom_feats)

        x = x[kept]

        assert x.shape[0] == geom_feats.shape[0]

        x = x[indices]

        """ import pickle
        with open("precomputed_features.pkl", "rb") as f:
            data = pickle.load(f) """

        x = bev_pool(x, geom_feats, ranks, B, self.nx[2], self.nx[0], self.nx[1], self.training)

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def bev_pool_precomputed(self, x, geom_feats, kept, ranks, indices):

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        x = x[kept]
        assert x.shape[0] == geom_feats.shape[0]

        x = x[indices]
        x = bev_pool(x, geom_feats, ranks, B, self.nx[2], self.nx[0], self.nx[1], self.training)

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

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
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        if geom_feats_precomputed is not None:
            geom_feats, kept, ranks, indices = geom_feats_precomputed
            x = self.get_cam_feats(img)
            x = self.bev_pool_precomputed(x, geom_feats, kept, ranks, indices)

        else:

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
            x = self.get_cam_feats(img)
            x = self.bev_pool(x, geom)

        return x


@MODELS.register_module()
class LSSTransform(BaseViewTransform):

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
    ) -> None:
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
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


class BaseDepthTransform(BaseViewTransform):

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        camera_intrinsics_inverse,
        img_aug_matrix_inverse,
        lidar_aug_matrix_inverse,
        geom_feats_precomputed,
    ):
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        if lidar_aug_matrix_inverse is None:
            lidar_aug_matrix_inverse = torch.inverse(lidar_aug_matrix)

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)
        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = lidar_aug_matrix_inverse[b, :3, :3].matmul(cur_coords.transpose(1, 0))
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )

            # NOTE(knzo25): in the original code, a per-image loop was
            # implemented to compute the depth. However, it fixes the number
            # of images, which is not desired for deployment (the number
            # of images may change due to frame drops).
            # For this reason, I modified the code to use tensor operations,
            # but the results will change due to indexing having potential
            # duplicates !. In practce, only about 0.01% of the elements will
            # have different results...

            indices = torch.nonzero(on_img, as_tuple=False)
            camera_indices = indices[:, 0]
            point_indices = indices[:, 1]

            masked_coords = cur_coords[camera_indices, point_indices].long()
            masked_dist = dist[camera_indices, point_indices]
            depth = depth.to(masked_dist.dtype)
            batch_size, num_imgs, channels, height, width = depth.shape
            # Depth tensor should have only one channel in this implementation
            assert channels == 1

            depth_flat = depth.view(batch_size, num_imgs, channels, -1)

            flattened_indices = camera_indices * height * width + masked_coords[:, 0] * width + masked_coords[:, 1]
            updates_flat = torch.zeros((num_imgs * channels * height * width), device=depth.device)

            updates_flat.scatter_(dim=0, index=flattened_indices, src=masked_dist)

            depth_flat[b] = updates_flat.view(num_imgs, channels, height * width)

            depth = depth_flat.view(batch_size, num_imgs, channels, height, width)

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        if geom_feats_precomputed is not None:
            # In inference, the geom_feats are precomputed
            geom_feats, kept, ranks, indices, camera_mask = geom_feats_precomputed
            x, est_depth_distr, gt_depth_distr, counts_3d = self.get_cam_feats(img, depth)

            """ data = {}
            data["img"] = img.cpu()
            data["depth"] = depth.cpu()
            data["x"] = x.cpu()
            import pickle
            with open("depth_deploy.pkl", "wb") as f:
                pickle.dump(data, f) """

            # At inference, if a camera is missing, we just mask the features
            camera_mask = camera_mask.view(1, -1, 1, 1, 1, 1)

            x = self.bev_pool_precomputed(x * camera_mask, geom_feats, kept, ranks, indices)
        else:
            intrins_inverse = torch.inverse(cam_intrinsic)[..., :3, :3]
            post_rots_inverse = torch.inverse(img_aug_matrix)[..., :3, :3]

            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins_inverse,
                post_rots_inverse,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )

            # Load from the pkl
            """ import pickle
            with open("precomputed_features.pkl", "rb") as f:
                data = pickle.load(f) """

            x, est_depth_distr, gt_depth_distr, counts_3d = self.get_cam_feats(img, depth)

            """ import pickle
            with open("depth_deploy.pkl", "rb") as f:
                data = pickle.load(f) """

            x = self.bev_pool(x, geom)

        if self.training:
            """counts_3d_aux = counts_3d.permute(0,1,4,2,3).unsqueeze(-1)
            gt_feats = gt_depth_distr.permute(0,1,4,2,3).unsqueeze(-1) * (counts_3d_aux > 0).float()
            est_feats = est_depth_distr.permute(0,1,4,2,3).unsqueeze(-1)

            num_cameras = gt_feats.shape[1]

            gt_bev_feats = self.bev_pool_precomputed(gt_feats, geom_feats, kept, ranks, indices)
            est_bev_feats = self.bev_pool_precomputed(est_feats, geom_feats, kept, ranks, indices)

            import pickle
            data = {}
            data["gt_bev_feats"] = gt_bev_feats.cpu().numpy()
            data["est_bev_feats"] = est_bev_feats.cpu().numpy()

            for i in range(num_cameras):
                gt_feats_aux = torch.zeros_like(gt_feats)
                gt_feats_aux[:,i] = gt_feats[:,i]
                gt_bev_feats_aux = self.bev_pool_precomputed(gt_feats_aux, geom_feats, kept, ranks, indices)

                est_feats_aux = torch.zeros_like(est_feats)
                est_feats_aux[:,i] = est_feats[:,i]
                est_bev_feats_aux = self.bev_pool_precomputed(est_feats_aux, geom_feats, kept, ranks, indices)

                data[f"gt_bev_feats_{i}"] = gt_bev_feats_aux.cpu().numpy()
                data[f"est_bev_feats_{i}"] = est_bev_feats_aux.cpu().numpy()

            with open("bev_features.pkl", "wb") as f:
                pickle.dump(data, f)"""

            mask_flat = counts_3d.sum(dim=-1).view(-1) > 0

            gt_depth_distr_flat = gt_depth_distr.view(-1, self.D)
            est_depth_distr_flat = est_depth_distr.reshape(-1, self.D)

            cross_ent = -torch.sum(gt_depth_distr_flat * torch.log(est_depth_distr_flat + 1e-8), dim=-1)
            cross_ent_masked = cross_ent * mask_flat.float()
            depth_loss = torch.sum(cross_ent_masked) / (mask_flat.sum() + 1e-8)
        else:
            depth_loss = 0.0

        return x, depth_loss


@MODELS.register_module()
class DepthLSSTransform(BaseDepthTransform):

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
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
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
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape
        h, w = self.image_size
        BN = B * N

        d = d.view(BN, *d.shape[2:])
        x = x.view(BN, C, fH, fW)

        # =================== TEST
        if self.training or True:
            camera_id = torch.arange(BN).view(-1, 1, 1).expand(BN, h, w)
            rows = torch.arange(h).view(1, -1, 1).expand(BN, h, w)
            cols = torch.arange(w).view(1, 1, -1).expand(BN, h, w)

            cell_j = rows // (h // fH)
            cell_i = cols // (w // fW)

            cell_id = camera_id * fH * fW + cell_j * fW + cell_i
            cell_id = cell_id.to(device=d.device)

            dist_bins = (
                d.clamp(min=self.dbound[0], max=self.dbound[1] - 0.5 * self.dbound[2])
                + 0.5 * self.dbound[2]
                - self.dbound[0]
            ) / self.dbound[2]
            dist_bins = dist_bins.long()

            flat_cell_id = cell_id.view(-1)
            flat_dist_bin = dist_bins.view(-1)

            flat_index = flat_cell_id * self.D + flat_dist_bin

            counts_flat = torch.zeros(BN * fH * fW * self.D, dtype=torch.float, device=d.device)
            counts_flat.scatter_add_(
                0, flat_index, torch.ones_like(flat_index, dtype=torch.float, device=flat_index.device)
            )

            counts_3d = counts_flat.view(B, N, fH, fW, self.D)
            counts_3d[..., 0] = 0.0

            # mask_flat = counts_3d.sum(dim=-1).view(-1) > 0

            # gt_depth_distr = torch.softmax(counts_3d, dim=-1)
            gt_depth_distr = counts_3d / (counts_3d.sum(dim=-1, keepdim=True) + 1e-8)
            # gt_depth_distr_flat = gt_depth_distr.view(-1, self.D)
            # =================== TEST
        else:
            gt_depth_distr = None
            counts_3d = None

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        est_depth_distr = depth.permute(0, 2, 3, 1).reshape(B, N, fH, fW, self.D)

        if self.training:
            depth_aux = gt_depth_distr.view(B * N, fH, fW, self.D).permute(0, 3, 1, 2)
            depth = depth + (torch.maximum(depth_aux, depth) - depth).detach()
        # Need to match the (B, N, H, W, D) order

        # est_depth_distr_flat = est_depth_distr.reshape(-1, self.D)

        """ import pickle
        data = {}
        data["gt_depth"] = gt_depth_distr.cpu().numpy()
        data["estimated_depth"] = est_depth_distr.cpu().numpy()
        data["counts"] = counts_3d.cpu().numpy()
        with open("estimated_depth.pkl", "wb") as f:
            pickle.dump(data, f) """

        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, est_depth_distr, gt_depth_distr, counts_3d

    def forward(self, *args, **kwargs):
        x, depth_loss = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x, depth_loss
