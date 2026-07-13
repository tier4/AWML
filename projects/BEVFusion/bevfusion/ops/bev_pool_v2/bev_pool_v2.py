# Copyright (c) Phigent Robotics. All rights reserved.

import numpy as np
import torch

from . import bev_pool_v2_ext


class QuickCumsumV2TrainingCuda(torch.autograd.Function):
    r"""BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.

    Please refer to the `paper <https://arxiv.org/abs/2211.17111>`_
    """

    @staticmethod
    def forward(
        ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, interval_starts, interval_lengths
    ):
        ranks_bev = ranks_bev.int()
        depth = depth.contiguous().float()
        feat = feat.contiguous().float()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        out = feat.new_zeros(bev_feat_shape)

        bev_pool_v2_ext.bev_pool_v2_forward(
            depth,
            feat,
            out,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors

        order = ranks_feat.argsort()
        ranks_feat, ranks_depth, ranks_bev = ranks_feat[order], ranks_depth[order], ranks_bev[order]
        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = torch.where(kept)[0].int()
        interval_lengths_bp = torch.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths_bp = interval_lengths_bp.contiguous()
        interval_starts_bp = interval_starts_bp.contiguous()

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        out_grad = out_grad.contiguous()
        bev_pool_v2_ext.bev_pool_v2_backward(
            out_grad,
            depth_grad,
            feat_grad,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths_bp,
            interval_starts_bp,
        )
        return depth_grad, feat_grad, None, None, None, None, None, None, None, None


class QuickCumsumV2Cuda(torch.autograd.Function):

    @staticmethod
    def symbolic(
        g,
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        out_height=128,
        out_width=128,
    ):
        """symbolic function for creating onnx op."""
        x = g.op(
            "autoware::QuickCumsumV2Cuda",
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            out_height_i=out_height,
            out_width_i=out_width,
        )

        # features_shape = _get_tensor_sizes(feat)
        # if features_shape is not None and hasattr(x.type(), "with_sizes"):
        #     output_type = x.type().with_sizes([B, D, H, W, _get_tensor_dim_size(x, -1)])
        #     output.setType(output_type)

    @staticmethod
    def forward(
        ctx,
        depth,  # B,N,D,H,W
        feat,  # B,N,H,W,C
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        out_height=128,
        out_width=128,
    ):
        """run forward."""
        out = feat.new_zeros(depth.shape[0], 1, out_height, out_width, feat.shape[-1])
        bev_feat = bev_pool_v2_ext.bev_pool_v2_forward(
            depth,
            feat,
            out,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )
        return bev_feat

    @staticmethod
    def backward(ctx, out_grad):
        raise NotImplementedError


def bev_pool_v2(
    depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, bev_feat_shape, is_training
):
    # Always use full (B, Z, H, W, C) buffer; QuickCumsumV2Cuda (Z=1) is ONNX-only.
    del is_training
    x = QuickCumsumV2TrainingCuda.apply(
        depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, interval_starts, interval_lengths
    )

    # Final shape: (B, C, Z, H, W) — matches LSSTransform v1 after permute
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


def test_bev_pool_v2():
    depth = np.array([0.3, 0.4, 0.2, 0.1, 0.7, 0.6, 0.8, 0.9])
    depth = torch.from_numpy(depth).float().cuda()
    depth = depth.view(1, 1, 2, 2, 2).requires_grad_()
    feat = torch.ones(size=[1, 1, 2, 2, 2], dtype=torch.float, device="cuda").requires_grad_()
    ranks_depth = torch.from_numpy(np.array([0, 4, 1, 6])).int().cuda()
    ranks_feat = torch.from_numpy(np.array([0, 0, 1, 2])).int().cuda()
    ranks_bev = torch.from_numpy(np.array([0, 0, 1, 1])).int().cuda()

    kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    if len(interval_starts) == 0:
        return None, None, None, None, None
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
    bev_feat = bev_pool_v2(
        depth, feat, ranks_depth, ranks_feat, ranks_bev, (1, 1, 2, 2, 2), interval_starts, interval_lengths
    )
    loss = torch.sum(bev_feat)
    loss.backward()
    assert loss == 4.4
    grad_depth = np.array([2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    grad_depth = torch.from_numpy(grad_depth).float()
    grad_depth = grad_depth.cuda().view(1, 1, 2, 2, 2)
    assert depth.grad.allclose(grad_depth)
    grad_feat = np.array([1.0, 1.0, 0.4, 0.4, 0.8, 0.8, 0.0, 0.0])
    grad_feat = torch.from_numpy(grad_feat).float().cuda().view(1, 1, 2, 2, 2)
    assert feat.grad.allclose(grad_feat)
