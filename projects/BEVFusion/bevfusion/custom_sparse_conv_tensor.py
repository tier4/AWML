"""
Custom SparseConvTensor for BEVFusion.
This customiztion is used to support cleaner ONNX export of sparse convolutions.
"""

from typing import Union, List, Optional

import torch
from spconv.pytorch import SparseConvTensor
from spconv.core import ConvAlgo


class CustomSparseConvTensor(SparseConvTensor):
    def __init__(self,
                 features: torch.Tensor,
                 indices: torch.Tensor,
                 spatial_shape: Union[List[int], np.ndarray],
                 batch_size: int,
                 grid: Optional[torch.Tensor] = None,
                 voxel_num: Optional[torch.Tensor] = None,
                 indice_dict: Optional[dict] = None,
                 benchmark: bool = False,
                 permanent_thrust_allocator: bool = False,
                 enable_timer: bool = False,
                 force_algo: Optional[ConvAlgo] = None):
      """
      Check the superclass documentation for more details.
      """
      
      super().__init__(
        features=features, 
        indices=indices, 
        spatial_shape=spatial_shape, 
        batch_size=batch_size, 
        grid=grid, 
        voxel_num=voxel_num, 
        indice_dict=indice_dict, 
        benchmark=benchmark, 
        permanent_thrust_allocator=permanent_thrust_allocator, 
        enable_timer=enable_timer, 
        force_algo=force_algo)
        
      # Precomputation for dense output shape.
      self.spatial_shape_list = list(self.spatial_shape)
      self.spatial_ndim = len(self.spatial_shape_list)
      self.trans_params = list(range(0, self.spatial_ndim + 1))
      self.trans_params.insert(1, self.spatial_ndim + 1)

    def dense(self, channels_first: bool = True):
        """
        Convert the sparse tensor to a dense tensor.
        """
        C = self.features.shape[1]
        out = self.features.zeros(
            [
                self.batch_size,
                *self.spatial_shape_list,
                C,
            ]
        )
        idx = self.indices.to(self.features.device).long()  # [N, 1+D]
        out.index_put_(idx.unbind(1), self.features)
        if not channels_first:
            return out 
        
        out = out.permute(*self.trans_params).contiguous()
        return out
 