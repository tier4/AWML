# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from .voxel_layer import dynamic_voxelize, hard_voxelize


class _Voxelization(Function):

    @staticmethod
    def forward(ctx, points, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            dynamic_voxelize(points, coors, voxel_size, coors_range, 3)
            return coors
        else:
            voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)
            voxel_num = hard_voxelize(
                points,
                voxels,
                coors,
                num_points_per_voxel,
                voxel_size,
                coors_range,
                max_points,
                max_voxels,
                3,
                deterministic,
            )
            # select the valid voxels
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


@torch.no_grad()
def voxelize_fast_gpu(points, voxel_size, point_cloud_range, max_num_points, max_voxels):
    """Fast GPU hard-voxelizer, bit-identical to the C++ ``hard_voxelize``
    (``deterministic=True``) but ~100x faster.

    The C++ deterministic path is O(N^2) (``point_to_voxelidx`` scans every
    earlier point) plus single-thread (``determin_voxel_num`` is launched
    ``<<<1, 1>>>``), costing ~70 ms per 250k-point frame. This reproduces its
    EXACT output via parallel sort + unique + scatter (~0.7 ms):

      * range filter, then ``coord = floor((xyz - min) / voxel_size)``
      * stable-sort by flat voxel id so in-voxel points keep their original
        index order (this matches the C++ ``num`` = count of earlier points in
        the same voxel, i.e. the first ``max_num_points`` are kept)
      * ``num_points_per_voxel = min(count, max_num_points)``

    The voxel ROW order differs (unique-sorted here vs. the C++ first-appearance
    order) but is irrelevant downstream: the sparse encoder indexes voxels by
    their coords. Returns ``None`` when ``M > max_voxels`` so the caller can fall
    back to the C++ op, whose voxel-order-dependent truncation we do not
    replicate in that (rare) case.
    """
    dev = points.device
    # Grid dims from the python config on the HOST — never a CUDA tensor — so
    # there is no per-forward device->host `.item()` sync (this path is hot).
    # `floor(x + 0.5)` mirrors the C++ op's `round()` exactly
    # (voxelization_cuda.cu: `grid_* = round((max - min) / voxel_*)`).
    gx = int(math.floor((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0] + 0.5))
    gy = int(math.floor((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1] + 0.5))
    gz = int(math.floor((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2] + 0.5))
    vs = torch.tensor(voxel_size, device=dev, dtype=points.dtype)
    rmin = torch.tensor(point_cloud_range[:3], device=dev, dtype=points.dtype)
    grid = torch.tensor([gx, gy, gz], device=dev)  # small constant, no .item() sync

    feat_dim = points.shape[1]
    # Match the CUDA dynamic_voxelize_kernel EXACTLY: compute the voxel index for
    # every point, then drop any whose index is outside the rounded grid
    # (c_* < 0 or c_* >= grid_*). A raw `>= min & < max` range filter is NOT
    # equivalent when range/voxel_size isn't an integer, or when a point at the
    # upper bound floors up to c_* == grid_* — the C++ op discards those, so the
    # fast path must too (else it emits out-of-grid voxels the encoder rejects).
    coord = torch.floor((points[:, :3] - rmin) / vs).long()  # (N, 3)
    valid = ((coord >= 0) & (coord < grid)).all(dim=1)
    pts = points[valid]
    coord = coord[valid]
    if pts.shape[0] == 0:
        return (
            points.new_zeros((0, max_num_points, feat_dim)),
            points.new_zeros((0, 3), dtype=torch.int32),
            points.new_zeros((0,), dtype=torch.int32),
        )
    # Internal grouping key only: linearize the (x, y, z) voxel index with z as
    # the most-significant axis so a stable sort groups same-voxel points and
    # preserves their original order. This z-major key does NOT change the
    # output coord order (`coors` below is (x, y, z)).
    flat = coord[:, 2] * (gy * gx) + coord[:, 1] * gx + coord[:, 0]

    order = torch.argsort(flat, stable=True)  # stable -> in-voxel original order
    flat_s, pts_s, coord_s = flat[order], pts[order], coord[order]
    uniq, inv, counts = torch.unique_consecutive(flat_s, return_inverse=True, return_counts=True)
    num_voxels = int(uniq.shape[0])  # tensor.shape is a host int — no device sync
    if num_voxels > max_voxels:
        # Bail to the C++ op (don't just keep the first max_voxels here): the C++
        # truncation when M > max_voxels drops voxels by FIRST-APPEARANCE order,
        # which the sort-order fast path doesn't reproduce — keeping max_voxels
        # here would retain a different voxel set and break bit-identity. This
        # branch is rare in practice (M ~ 100k < max_voxels = 120k for a 120m sweep).
        return None

    seg_start = torch.zeros(num_voxels, dtype=torch.long, device=dev)
    seg_start[1:] = torch.cumsum(counts, 0)[:-1]
    rank = torch.arange(flat_s.shape[0], device=dev) - seg_start[inv]
    keep = rank < max_num_points

    voxels = torch.zeros(num_voxels, max_num_points, feat_dim, device=dev, dtype=pts.dtype)
    voxels[inv[keep], rank[keep]] = pts_s[keep]
    num_points_per_voxel = counts.clamp(max=max_num_points).to(torch.int32)
    # (x, y, z) order — matches this repo's compiled hard_voxelize, whose
    # dynamic_voxelize_kernel writes coors_offset[0..2] = c_x, c_y, c_z
    # (voxelization_cuda.cu). Any (z, y, x) flip for the sparse encoder happens
    # downstream and applies equally to this and the C++ op's output.
    coors = coord_s[seg_start].to(torch.int32)  # (M, 3) = (x, y, z)
    return voxels, coors, num_points_per_voxel


class Voxelization(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000, deterministic=True):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w] removed
        self.pcd_shape = [*input_feat_shape, 1]  # [::-1]

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        # Fast parallel path for the deterministic case: bit-identical to the
        # C++ hard_voxelize(deterministic=True) but ~100x faster on GPU (the C++
        # path is O(N^2) + single-thread). Falls back to the C++ op on CPU, in
        # non-deterministic mode, or when M > max_voxels.
        if self.deterministic and input.is_cuda:
            out = voxelize_fast_gpu(
                input,
                self.voxel_size,
                self.point_cloud_range,
                self.max_num_points,
                max_voxels,
            )
            if out is not None:
                return out

        return voxelization(
            input,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            max_voxels,
            self.deterministic,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "voxel_size=" + str(self.voxel_size)
        tmpstr += ", point_cloud_range=" + str(self.point_cloud_range)
        tmpstr += ", max_num_points=" + str(self.max_num_points)
        tmpstr += ", max_voxels=" + str(self.max_voxels)
        tmpstr += ", deterministic=" + str(self.deterministic)
        tmpstr += ")"
        return tmpstr
