import torch
from torch.autograd import Function
from torch.onnx.symbolic_helper import _get_tensor_sizes

try:
    import pointrope_cuda as _kernels  # run `python setup.py install`
except ModuleNotFoundError:
    from . import pointrope_cuda as _kernels  # run `python setup.py build_ext --inplace`


class PointROPE_func(Function):
    @staticmethod
    def symbolic(g, tokens, positions, base, F0=1.0):
        output = g.op(
            "litept::PointRoPE",
            tokens,
            positions,
            base_f=float(base),
            f0_f=float(F0),
            outputs=1,
        )
        token_shape = _get_tensor_sizes(tokens)
        # PointRoPE is in-place in forward() and returns a tensor with the same shape as `tokens`.
        # Preserve the full [B, N, H, D] shape for ONNX shape inference when available.
        if token_shape is not None and hasattr(output.type(), "with_sizes"):
            output_type = tokens.type().with_sizes(token_shape)
            output.setType(output_type)

        return output

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1.0):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        # tokens = tokens.clone() # uncomment this if inplace doesn't work
        _kernels.pointrope(tokens, positions, base, F0)
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        _kernels.pointrope(grad_res, positions, base, -F0)
        ctx.mark_dirty(grad_res)
        return grad_res, None, None, None


# class PointROPE(torch.nn.Module):
#     def __init__(self, freq=100.0, F0=1.0):
#         super().__init__()
#         self.base = freq
#         self.F0 = F0

#     def forward(self, tokens, positions):
#         PointROPE_func.apply(tokens.transpose(1, 2), positions, self.base, self.F0)
#         return tokens


class PointROPE(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor):
        """"""
        tokens = tokens.transpose(1, 2)
        assert tokens.dim() == 4, tokens.shape
        B, N, H, D = tokens.shape
        assert D % 6 == 0
        Q = D // 6

        # pos -> [B, N, 3]
        if positions.dim() == 2:
            assert positions.shape[0] == B * N and positions.shape[1] == 3
            pos_bn3 = positions.view(B, N, 3)
        else:
            assert positions.shape == (B, N, 3), f"{positions.shape=} vs {tokens.shape=}"
            pos_bn3 = positions

        # inv_freq: [Q]
        # inv_freq[q] = fwd / base^(q/Q)
        q = torch.arange(Q, device=tokens.device, dtype=tokens.dtype)
        inv_freq = torch.tensor(float(self.F0), device=tokens.device, dtype=tokens.dtype) / (
            torch.tensor(float(self.base), device=tokens.device, dtype=tokens.dtype) ** (q / float(Q))
        )

        # freq: [B, N, 3, Q]
        pos_f = pos_bn3.to(dtype=tokens.dtype)
        freq = pos_f.unsqueeze(-1) * inv_freq.view(1, 1, 1, Q)

        cos = torch.cos(freq)  # [B, N, 3, Q]
        sin = torch.sin(freq)  # [B, N, 3, Q]

        # separate tokens into 6-blocks: each [B, N, H, Q]
        ux, vx, uy, vy, uz, vz = tokens.split(Q, dim=-1)

        # broadcast each [B, N, 1, Q] to head per axis
        cx = cos[:, :, 0, :].unsqueeze(2)
        sx = sin[:, :, 0, :].unsqueeze(2)
        cy = cos[:, :, 1, :].unsqueeze(2)
        sy = sin[:, :, 1, :].unsqueeze(2)
        cz = cos[:, :, 2, :].unsqueeze(2)
        sz = sin[:, :, 2, :].unsqueeze(2)

        ru_x = ux * cx - vx * sx
        rv_x = vx * cx + ux * sx

        ru_y = uy * cy - vy * sy
        rv_y = vy * cy + uy * sy

        ru_z = uz * cz - vz * sz
        rv_z = vz * cz + uz * sz

        return torch.cat([ru_x, rv_x, ru_y, rv_y, ru_z, rv_z], dim=-1).transpose(1, 2)
