import torch
from torch.onnx.symbolic_helper import _get_tensor_dim_size, _get_tensor_sizes

from . import bev_pool_ext


class QuickCumsum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickCumsumTrainingCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None


class QuickCumsumCuda(torch.autograd.Function):

    @staticmethod
    def symbolic(
        g,
        x,
        geom_feats,
        interval_lengths,
        interval_starts,
        B,
        D,
        H,
        W,
    ):
        output = g.op(
            "autoware::QuickCumsumCuda",
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            batch_size_i=B,
            dimension_i=D,
            height_i=H,
            width_i=W,
            outputs=1,
        )

        features_shape = _get_tensor_sizes(x)
        if features_shape is not None and hasattr(x.type(), "with_sizes"):
            output_type = x.type().with_sizes([B, D, H, W, _get_tensor_dim_size(x, -1)])
            output.setType(output_type)

        return output

    @staticmethod
    def forward(ctx, x, geom_feats, interval_lengths, interval_starts, B, D, H, W):
        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )
        return out

    @staticmethod
    def backward(ctx, out_grad):
        raise NotImplementedError


def bev_pool(feats, coords, ranks, B, D, H, W, is_training):
    assert feats.shape[0] == coords.shape[0]

    # NOTE(knzo25): we want to put all the operations we can in the graph
    if is_training:
        x = QuickCumsumTrainingCuda.apply(feats, coords, ranks, B, D, H, W)

    else:

        kept = torch.ones(feats.shape[0], device=feats.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = feats.shape[0] - interval_starts[-1]

        if coords.dtype != torch.int32:
            coords = coords.int()

        x = QuickCumsumCuda.apply(
            feats, coords, interval_lengths, interval_starts, int(B), D.item(), H.item(), W.item()
        )

    x = x.permute(0, 4, 1, 2, 3).contiguous()

    return x
