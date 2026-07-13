"""Fuse ``SparseConvolution + BatchNorm1d`` in a BEVFusion sparse encoder.

Plain graph optimization used by the FP16 deployment export path: fold ``Conv -> BN``
into the conv weights (eval mode) so the exported sparse ONNX is BN-free and matches the
runtime graph. This is *not* quantization — it only rewrites the module tree.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def fuse_spconv_bn_in_encoder(sparse_encoder: nn.Module) -> int:
    """Fuse each ``SparseConvolution`` + ``BatchNorm1d`` pair in ``sparse_encoder`` (eval mode).

    Returns the number of fused Conv-BN pairs.
    """
    try:
        # spconv ships an eval-mode Conv+BN fold helper (library utility, not INT8).
        from spconv.pytorch.quantization.utils import fuse_spconv_bn_eval
    except ImportError:
        logger.warning("spconv BN-fusion helper not available; skipping SparseConv+BN fusion")
        return 0

    from spconv.pytorch.conv import SparseConvolution

    sparse_encoder.eval()
    fused_count = 0

    for module in sparse_encoder.modules():
        children = list(module._modules.items())
        for i in range(len(children) - 1):
            left_name, left_mod = children[i]
            right_name, right_mod = children[i + 1]
            if isinstance(left_mod, SparseConvolution) and isinstance(right_mod, torch.nn.BatchNorm1d):
                fused_conv = fuse_spconv_bn_eval(left_mod, right_mod)
                setattr(module, left_name, fused_conv)
                setattr(module, right_name, torch.nn.Identity())
                fused_count += 1

    return fused_count
