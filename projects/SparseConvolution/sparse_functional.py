from typing import Any, List, Optional

import numpy as np
import torch
from cumm import tensorview as tv
from spconv import constants
from spconv.algo import CONV_CPP
from spconv.constants import SPCONV_DO_SORT, SPCONV_USE_DIRECT_TABLE, AllocKeys
from spconv.core import ConvAlgo
from spconv.core_cc.csrc.sparse.all import SpconvOps
from spconv.core_cc.csrc.sparse.convops.spops import ConvGemmOps
from spconv.pytorch.core import ThrustSortAllocator
from spconv.pytorch.cppcore import _TORCH_DTYPE_TO_TV, TorchAllocator, get_arch, get_current_stream, torch_tensor_to_tv
from spconv.tools import CUDAKernelTimer
from torch.autograd import Function
from torch.onnx.symbolic_helper import _get_tensor_sizes


class GetIndicePairsImplicitGemm(Function):

    @staticmethod
    def symbolic(
        g,
        indices: torch.Tensor,
        batch_size: int,
        spatial_shape: List[int],
        algo: ConvAlgo,
        ksize: List[int],
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        out_padding: List[int],
        subm: bool,
        transpose: bool,
        is_train: bool,
        alloc: Optional[ThrustSortAllocator],
        timer: CUDAKernelTimer,
    ):
        outputs = g.op(
            "autoware::GetIndicePairsImplicitGemm",
            indices,
            batch_size_i=batch_size,
            spatial_shape_i=spatial_shape,
            algo_i=algo.value,
            ksize_i=ksize,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            out_padding_i=out_padding,
            subm_i=subm,
            transpose_i=transpose,
            is_train_i=is_train,
            outputs=5,
        )
        indices_shape = _get_tensor_sizes(indices)
        if indices_shape is not None and hasattr(indices.type(), "with_sizes"):
            output_type_1 = indices.type().with_sizes([None, indices_shape[1]])
            output_type_2 = indices.type().with_sizes([np.prod(ksize), None])
            output_type_3 = indices.type().with_sizes([None, 1])
            output_type_4 = indices.type().with_sizes([None])
            output_type_5 = indices.type().with_sizes([])

            outputs[0].setType(output_type_1)
            outputs[1].setType(output_type_2)
            outputs[2].setType(output_type_3)
            outputs[3].setType(output_type_4)
            outputs[4].setType(output_type_5)
        return outputs

    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        batch_size: int,
        spatial_shape: List[int],
        algo: ConvAlgo,
        ksize: List[int],
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        out_padding: List[int],
        subm: bool,
        transpose: bool,
        is_train: bool,
        alloc: Optional[ThrustSortAllocator],
        timer: CUDAKernelTimer,
    ) -> torch.Tensor:
        """Why return tuple?

        because pytorch seems don't support custom object in autograd.
        return: (
            out_inds,
            num_inds_per_loc,
            pair_fwd,
            pair_bwd, # torch.Tensor() if subm or inference mode
            pair_mask_fwd_splits,
            pair_mask_bwd_splits, # torch.Tensor() if subm or inference mode
            mask_argsort_fwd_splits,
            mask_argsort_bwd_splits, # torch.Tensor() if subm or inference mode
            masks,
        )
        direct_table: a hash-based regular conv pair gen algo
        to avoid unique operation.
        runs faster than pytorch unique with num_voxel < 1000k.
        """

        num_out_act_bound: int = -1
        direct_table: bool = SPCONV_USE_DIRECT_TABLE
        do_sort = SPCONV_DO_SORT

        stream = get_current_stream()

        thalloc = TorchAllocator(indices.device)
        timer_cpp = tv.CUDAKernelTimer(False)
        if timer._timer is not None:
            timer_cpp = timer._timer

        mask_tensor, num_act_out = SpconvOps.get_indice_pairs_implicit_gemm(
            thalloc,
            torch_tensor_to_tv(indices),
            batch_size,
            spatial_shape,
            algo.value,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm,
            transpose,
            is_train,
            stream,
            num_out_act_bound,
            timer=timer_cpp,
            direct_table=direct_table,
            do_sort=do_sort,
        )

        mask_split_count = mask_tensor.dim(0)
        # NOTE(knzo25): we support only the simplest case
        assert mask_split_count == 1
        if subm:
            out_inds = indices
        else:
            out_inds = thalloc.allocated[AllocKeys.OutIndices]

        if subm:
            pair = thalloc.allocated[AllocKeys.PairFwd]
            pair_mask = thalloc.allocated[AllocKeys.PairMask]
            mask_argsort = thalloc.allocated[AllocKeys.MaskArgSort]
            pair_mask_in_splits = pair_mask[0]
            mask_argsort_in_splits = mask_argsort[0]
            pair_fwd = pair[0]
            return (out_inds, pair[0], pair_mask_in_splits, mask_argsort_in_splits, num_act_out)
        else:
            pair_fwd = thalloc.allocated[AllocKeys.PairFwd]
            pair_mask_fwd = thalloc.allocated[AllocKeys.PairMask]
            mask_argsort_fwd = thalloc.allocated[AllocKeys.MaskArgSort]
            pair_mask_fwd_splits = pair_mask_fwd[0]
            mask_argsort_fwd_splits = mask_argsort_fwd[0]

            return (out_inds, pair_fwd, pair_mask_fwd_splits, mask_argsort_fwd_splits, num_act_out)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        return None, None, None, None, None, None, None, None, None, None


class ImplicitGemm(Function):

    @staticmethod
    def symbolic(
        g,
        features: torch.Tensor,
        filters: torch.Tensor,
        pair_fwd: torch.Tensor,
        pair_mask_fwd_splits: torch.Tensor,
        mask_argsort_fwd_splits: torch.Tensor,
        num_activate_out: int,
        masks: List[np.ndarray],
        is_train: bool,
        is_subm: bool,
        timer: CUDAKernelTimer,
        fp32_accum: Optional[bool],
        bias: Optional[torch.Tensor],
        act_alpha: float,
        act_beta: float,
        act_type: tv.gemm.Activation,
        output_scale: float,
        scale: Optional[torch.Tensor],
        output_add: Optional[torch.Tensor],
        output_add_scale: float,
        output_dtype: Optional[torch.dtype],
    ):

        output = g.op(
            "autoware::ImplicitGemm",
            features,
            filters,
            pair_fwd,
            pair_mask_fwd_splits,
            mask_argsort_fwd_splits,
            is_train_i=is_train,
            is_subm_i=is_subm,
            fp32_accum_i=fp32_accum,
            act_alpha_f=act_alpha,
            act_beta_f=act_beta,
            output_scale_f=output_scale,
            output_add_scale_f=output_add_scale,
            outputs=1,
        )
        features_shape = _get_tensor_sizes(features)
        filters_shape = _get_tensor_sizes(filters)
        if features_shape is not None and hasattr(features.type(), "with_sizes"):
            output_type = features.type().with_sizes([features_shape[0], filters_shape[0]])
            output.setType(output_type)

        return output

    @staticmethod
    def forward(
        ctx,
        features: torch.Tensor,
        filters: torch.Tensor,
        pair_fwd: torch.Tensor,
        pair_mask_fwd_splits: torch.Tensor,
        mask_argsort_fwd_splits: torch.Tensor,
        num_activate_out: int,
        masks: List[np.ndarray],
        is_train: bool,
        is_subm: bool,
        timer: CUDAKernelTimer = CUDAKernelTimer(False),
        fp32_accum: Optional[bool] = None,
        bias: Optional[torch.Tensor] = None,
        act_alpha: float = 0.0,
        act_beta: float = 0.0,
        act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
        output_scale: float = 1.0,
        scale: Optional[torch.Tensor] = None,
        output_add: Optional[torch.Tensor] = None,
        output_add_scale: float = 0.0,
        output_dtype: Optional[torch.dtype] = None,
    ):

        # NOTE(knzo25): start of custom changes needed for deployment
        pair_mask_fwd_splits = [pair_mask_fwd_splits]
        mask_argsort_fwd_splits = [mask_argsort_fwd_splits]

        assert fp32_accum is None, "fp32_accum is not supported"
        assert bias is None, "bias is not supported"
        assert scale is None
        assert output_add is None
        assert output_dtype is torch.float32

        # NOTE(knzo25): end of custom changes needed for deployment

        stream = get_current_stream()
        bias_tv = tv.Tensor()
        scale_tv = tv.Tensor()
        output_add_tv = tv.Tensor()

        if not features.is_contiguous():
            features = features.contiguous()
        assert features.is_contiguous()
        assert filters.is_contiguous()
        if output_dtype is None:
            output_dtype = features.dtype

        alloc = TorchAllocator(features.device, features.dtype == torch.qint8)
        features_tv = torch_tensor_to_tv(features)
        pair_fwd_tv = torch_tensor_to_tv(pair_fwd)
        pair_mask_fwd_splits_tv = [torch_tensor_to_tv(t, tv.uint32) for t in pair_mask_fwd_splits]
        mask_argsort_fwd_splits_tv = [torch_tensor_to_tv(t) for t in mask_argsort_fwd_splits]

        filters_tv = torch_tensor_to_tv(filters)
        mask = np.array([np.iinfo(np.uint32).max], dtype=np.uint32)
        mask_tv = tv.from_numpy(mask).clone()
        timer_cpp = tv.CUDAKernelTimer(False)
        if timer._timer is not None:
            timer_cpp = timer._timer
        auto_fp32_accum = fp32_accum is None
        if fp32_accum is None:
            fp32_accum = False
        arch = get_arch()
        output_dtype_tv = _TORCH_DTYPE_TO_TV[output_dtype]

        _, _ = ConvGemmOps.implicit_gemm(
            alloc,
            CONV_CPP,
            features_tv,
            filters_tv,
            pair_fwd_tv,
            pair_mask_fwd_splits_tv,
            mask_argsort_fwd_splits_tv,
            num_activate_out,
            mask_tv,
            arch,
            is_train,
            is_subm,
            stream,
            timer_cpp,
            auto_fp32_accum,
            fp32_accum,
            bias_tv,
            act_alpha,
            act_beta,
            act_type,
            use_tf32=constants.SPCONV_ALLOW_TF32,
            output_scale=output_scale,
            scale=scale_tv,
            output_add=output_add_tv,
            output_add_scale=output_add_scale,
            output_dtype=output_dtype_tv,
        )
        out_features = alloc.allocated[AllocKeys.OutFeatures]
        mask_output_fwd = alloc.allocated.get(AllocKeys.MaskOutputFwd, None)
        if is_train:
            assert mask_output_fwd is not None

        return out_features


get_indice_pairs_implicit_gemm = GetIndicePairsImplicitGemm.apply
implicit_gemm = ImplicitGemm.apply
