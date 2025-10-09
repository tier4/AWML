"""
General Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch


@torch.inference_mode()
def offset2bincount(offset):
    # NOTE(knzo25): hack to avoid unsupported ops in export mode
    if len(offset) == 1:
        return offset
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.inference_mode()
def offset2batch(offset, coords=None):

    # NOTE(knzo25): hack to avoid unsupported ops in export mode
    if offset.size(0) == 1 and coords is not None:
        return torch.zeros((coords.shape[0]), device=coords.device, dtype=torch.long)

    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    # NOTE(knzo25): hack to avoid unsupported ops in export mode
    if batch.detach().cpu().numpy().max() == 0:
        return torch.Tensor([batch.size(0)]).to(device=batch.device).type(batch.dtype)
    return torch.cumsum(batch.bincount(), dim=0).long()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
