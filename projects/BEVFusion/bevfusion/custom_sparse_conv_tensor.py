"""
Custom SparseConvTensor for BEVFusion.
This customiztion is used to support cleaner ONNX export of sparse convolutions.
"""

import torch
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


def sparse_to_dense(sparse_tensor: SparseConvTensor, batch_size: int, spatial_shapes: list[int], out_channels: int):
    """
    Convert the sparse tensor to a dense tensor.
    """
    H, W, D = spatial_shapes
    num_cells = batch_size * H * W * D
    idx = sparse_tensor.indices.to(sparse_tensor.features.device).long()  # [N, 1+D]
    b, h, w, d = idx.unbind(1)
    # b * (H * W * D) + h*(W*D) + w*D + d
    # Factor out the common terms D and W
    # (b*H*W + h*W + w) * D + d -> (b*H + h) * W + w) * D + d
    linear_idx = ((b * H + h) * W + w) * D + d  # [N]

    out = torch.zeros(
        [num_cells, sparse_tensor.features.shape[1]],
        device=sparse_tensor.features.device,
        dtype=sparse_tensor.features.dtype,
    )
    scatter_idx = linear_idx.unsqueeze(1).expand(-1, out_channels)  # [N, C]
    out = out.scatter(0, scatter_idx, sparse_tensor.features)
    return out.view(batch_size, H, W, D, out_channels)
