# FastViT for PyTorch
#
# Original implementation and weights from https://github.com/apple/ml-fastvit
#
# For licensing see accompanying LICENSE file at https://github.com/apple/ml-fastvit/tree/main
# Original work is copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from mmdet3d.registry import MODELS
import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from typing import List, Sequence
from timm import create_model
from timm.models.fastvit import RepConditionalPosEnc,FastVit
from functools import partial

@MODELS.register_module(force=True)
class FastVit(nn.Module):
    """FastVit backbone for feature extraction.
    
    This module wraps the FastVit model from timm and extracts multi-level features
    similar to ConvNext_PC architecture.
    
    Args:
        variant_type (str): FastVit variant type. Default: "fastvit_sa24"
        out_indices (list): Output feature indices. Default: [0, 1, 2, 3]
        pretrained (bool): Whether to load pretrained weights. Default: True
        frozen_stages (int): Number of frozen stages. Default: 0
    """
    
    def __init__(
        self,
        variant_type: str = "fastvit_sa24",
        input_channels: int = 32,
        out_indices: List[int] = [0, 1, 2],
        out_channels: List[int] = None,
        pretrained: bool = False,
        frozen_stages: int = 0,  
        **kwargs 
    ):
        super().__init__()
        
        self.variant_type = variant_type
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # Create the FastVit model
        self.model = create_model(variant_type, in_chans=input_channels, pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs)
        # Get feature info to understand the output channels
        self.feature_info = self.model.feature_info
        
        # Store output channels for each stage
        if out_channels is None:
            self.out_channels = [info['num_chs'] for info in self.feature_info]
        else:
            assert out_channels== [self.feature_info[i]['num_chs'] for i in out_indices], "out_channels mismatch"
            self.out_channels = out_channels
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze parameters of specified stages."""
        if self.frozen_stages > 0:
            for name, module in self.model.named_children():
                if f"stages_{self.frozen_stages}" in name:
                  break
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        """Forward pass through FastVit backbone.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W)
            
        Returns:
            tuple: Multi-level feature maps
        """
        # Extract features using timm's features_only mode
        features = self.model(x)
        return tuple(features)
    
    def train(self, mode=True):
        """Set training mode and freeze stages."""
        super().train(mode)
        self._freeze_stages()
        return self


@MODELS.register_module(force=True)  
class FastVitSA24(FastVit):
    """FastVit SA24 variant with predefined configuration."""
    
    def __init__(self, **kwargs):
        model_args = dict(
            layers=(4, 4, 12),
            embed_dims=(64, 128, 256),
            mlp_ratios=(4, 4, 4),
            pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
            token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
            variant_type='fastvit_sa24'
        )
        kwargs.update(model_args)
        super().__init__(**kwargs)



@MODELS.register_module(force=True)
class FastVitSA36(FastVit):
    """FastVit SA36 variant with predefined configuration."""
    
    def __init__(self, **kwargs):
        model_args = dict(
            layers=(6, 6, 18),
            embed_dims=(64, 128, 256),
            mlp_ratios=(4, 4, 4),
            pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
            token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
            variant_type='fastvit_sa36'
        )
        kwargs.update(model_args)
        super().__init__(**kwargs)

@MODELS.register_module(force=True)
class FastVitMA36(FastVit):
    """FastVit MA36 variant with predefined configuration."""
    
    def __init__(self, **kwargs):
        model_args = dict(
            layers=(6, 6, 18),
            embed_dims=(76, 152, 304),
            mlp_ratios=(4, 4, 4),
            pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
            token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
            variant_type='fastvit_ma36'
        )
        kwargs.update(model_args)
        super().__init__(**kwargs)