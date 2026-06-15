"""
This file is used to write functions to deploy custom plugins to support Autoware, for example, TopK.
"""

import torch
from torch.autograd import Function
from torch.onnx.symbolic_helper import _get_tensor_sizes


class TopK(Function):

    @staticmethod
    def symbolic(
        g,
        x: torch.Tensor,
        k: int,
        dim: int,
        sorted: bool = False,
    ):

        output = g.op(
            "autoware::Argsort",
            x,
            outputs=1,
        )
        x_shape = _get_tensor_sizes(x)
        if x_shape is not None and hasattr(output.type(), "with_sizes"):
            output_type = x.type().with_sizes(x_shape)
            output.setType(output_type)
        # Argsort from Autoware is in ascending order, so we need to return the last k elements.
        return output[-k:]

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        k: int,
        dim: int,
        sorted: bool = False,
    ):
        _, indices = torch.topk(x, k=k, dim=dim, largest=True, sorted=sorted)
        return indices


def topk(x: torch.Tensor, k: int, dim: int, sorted: bool = False):
    return TopK.apply(x, k, dim, sorted)
