# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time
from typing import Optional

import torch
from cumm import tensorview as tv
from spconv.core import ConvAlgo
from spconv.debug_utils import spconv_save_debug_data
from spconv.pytorch import ops
from spconv.pytorch.conv import SparseConvolution as SparseConvolutionBase
from spconv.pytorch.core import ImplicitGemmIndiceData, SparseConvTensor
from spconv.utils import nullcontext
from torch.nn import functional as F

from . import sparse_functional as Fsp_custom

_MAX_NUM_VOXELS_DURING_TRAINING = "max_num_voxels_during_training"


def _apply_act(x: torch.Tensor, act_type: tv.gemm.Activation, act_alpha: float, act_beta: float):
    if act_type == tv.gemm.Activation.None_:
        return x
    elif act_type == tv.gemm.Activation.ReLU:
        return F.relu(x)
    elif act_type == tv.gemm.Activation.Sigmoid:
        return F.sigmoid(x)
    elif act_type == tv.gemm.Activation.LeakyReLU:
        return F.leaky_relu(x, act_alpha)
    else:
        raise NotImplementedError


class SparseConvolution(SparseConvolutionBase):

    def _conv_forward(
        self,
        training: bool,
        input: SparseConvTensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        add_input: Optional[SparseConvTensor] = None,
        channel_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[float] = None,
        name: Optional[str] = None,
        sparse_unique_name: str = "",
        act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
        act_alpha: float = 0,
        act_beta: float = 0,
    ):
        # assert isinstance(input, SparseConvTensor)
        is_int8 = input.is_quantized and weight.is_quantized
        if is_int8:
            raise NotImplementedError

        assert input.features.shape[1] == self.in_channels, "channel size mismatch"
        features = input.features
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        bias_for_training = bias if training else None
        bias_for_infer = bias if not training else None
        output_add_scale = 0.0
        if is_int8:
            raise NotImplementedError
        if training:
            raise NotImplementedError

        if not self.subm:
            if self.transposed:
                out_spatial_shape = ops.get_deconv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation, self.output_padding
                )
            else:
                out_spatial_shape = ops.get_conv_output_size(
                    spatial_shape, self.kernel_size, self.stride, self.padding, self.dilation
                )
        else:
            out_spatial_shape = spatial_shape
        # print(self._sparse_unique_name, spatial_shape, out_spatial_shape)
        # input.update_grid(out_spatial_shape)
        # t = time.time()
        out_tensor = input.shadow_copy()
        if input.benchmark:
            raise NotImplementedError

        if self.conv1x1 and not is_int8:
            raise NotImplementedError

        indice_dict = input.indice_dict.copy()
        # only support contiguous tensor for now
        if not features.is_contiguous():
            features = features.contiguous()
        algo = self.algo
        if self.indice_key is not None:
            data = input.find_indice_pair(self.indice_key)
            if data is not None:
                msg = "due to limitation of pytorch, " "you must provide same algo to " "layers share same indice key."
                assert algo == data.algo, msg
                # algo = data.algo
        profile_ctx = nullcontext()
        if input._timer is not None and sparse_unique_name:
            profile_ctx = input._timer.namespace(sparse_unique_name)
        with profile_ctx:
            if algo == ConvAlgo.Native:
                raise NotImplementedError
            else:
                data = input.find_indice_pair(self.indice_key)
                if data is not None:
                    assert isinstance(data, ImplicitGemmIndiceData)
                if self.inverse:
                    assert data is not None and self.indice_key is not None
                    assert data.is_subm is False, (
                        "inverse conv can only " "be used with standard " "conv and pool ops."
                    )
                    outids = data.indices
                    pair_fwd = data.pair_bwd
                    pair_bwd = data.pair_fwd
                    pair_mask_fwd_splits = data.pair_mask_bwd_splits
                    # pair_mask_bwd_splits = data.pair_mask_fwd_splits
                    mask_argsort_fwd_splits = data.mask_argsort_bwd_splits
                    mask_argsort_bwd_splits = data.mask_argsort_fwd_splits
                    masks = data.masks
                    out_spatial_shape = data.spatial_shape

                    self._check_inverse_reuse_valid(input, spatial_shape, data)
                else:
                    if self.indice_key is not None and data is not None:
                        outids = data.out_indices
                        pair_fwd = data.pair_fwd
                        pair_bwd = data.pair_bwd
                        pair_mask_fwd_splits = data.pair_mask_fwd_splits
                        # pair_mask_bwd_splits = data.pair_mask_bwd_splits
                        mask_argsort_fwd_splits = data.mask_argsort_fwd_splits
                        mask_argsort_bwd_splits = data.mask_argsort_bwd_splits
                        masks = data.masks
                        assert self.subm, "only support reuse subm indices"
                        self._check_subm_reuse_valid(input, spatial_shape, data)
                    else:
                        if input.benchmark:
                            torch.cuda.synchronize()
                            t = time.time()
                        with input._timer.namespace("gen_pairs"):
                            # we need to gen bwd indices for regular conv
                            # because it may be inversed.
                            try:
                                res = Fsp_custom.get_indice_pairs_implicit_gemm(
                                    indices,
                                    batch_size,
                                    spatial_shape,
                                    algo,
                                    self.kernel_size,
                                    self.stride,
                                    self.padding,
                                    self.dilation,
                                    self.output_padding,
                                    self.subm,
                                    self.transposed,
                                    (not self.subm) or training,
                                    input.thrust_allocator,
                                    input._timer,
                                )
                            except Exception as e:
                                msg = "[Exception|implicit_gemm_pair]"
                                msg += f"indices={indices.shape}," "bs={batch_size}," "ss={spatial_shape},"
                                msg += f"algo={algo}," "ksize={self.kernel_size}," "stride={self.stride},"
                                msg += f"padding={self.padding}," "dilation={self.dilation}," "subm={self.subm},"
                                msg += f"transpose={self.transposed}"
                                print(msg, file=sys.stderr)
                                spconv_save_debug_data(indices)
                                raise e
                        if input.benchmark:
                            torch.cuda.synchronize()
                            interval = time.time() - t
                            out_tensor.benchmark_record[name]["indice_gen_time"].append(interval)
                        outids = res[0]
                        # num_inds_per_loc = None  #res[1]
                        pair_fwd = res[1]  # res[2]
                        pair_bwd = None  # res[3]
                        pair_mask_fwd_splits = res[2]  # res[4]
                        pair_mask_bwd_splits = None  # res[5]
                        mask_argsort_fwd_splits = res[3]  # res[6]
                        mask_argsort_bwd_splits = None  # res[7]
                        # masks = res[8] we should not use this for test
                        masks = [None]
                        if self.indice_key is not None:
                            indice_data = ImplicitGemmIndiceData(
                                outids,
                                indices,
                                pair_fwd,
                                pair_bwd,
                                pair_mask_fwd_splits,
                                pair_mask_bwd_splits,
                                mask_argsort_fwd_splits,
                                mask_argsort_bwd_splits=mask_argsort_bwd_splits,
                                masks=masks,
                                is_subm=self.subm,
                                spatial_shape=spatial_shape,
                                out_spatial_shape=out_spatial_shape,
                                algo=algo,
                                ksize=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation,
                            )
                            msg = f"your indice key {self.indice_key} " "already exists in this sparse tensor."
                            assert self.indice_key not in indice_dict, msg
                            indice_dict[self.indice_key] = indice_data
                if input.benchmark:
                    torch.cuda.synchronize()
                    t = time.time()
                num_activate_out = outids.shape[0]  # TODO(knzo25): should use the output of res to force the graph
                weight_cur = weight
                bias_cur = bias_for_infer
                # if self.enable_int8_test_mode:
                #     assert features.dtype == torch.int8, ("in int8 "
                #         "test mode, feature must be int8")
                #     weight_cur = self._int8_weight
                #     bias_cur = self._int8_bias
                if training:
                    raise NotImplementedError
                else:
                    output_dtype = None
                    if output_scale is None:
                        output_dtype = weight.dtype
                    out_features = Fsp_custom.implicit_gemm(
                        features,
                        weight_cur,
                        pair_fwd,
                        pair_mask_fwd_splits,
                        mask_argsort_fwd_splits,
                        num_activate_out,
                        masks,
                        training,
                        self.subm,
                        input._timer,
                        self.fp32_accum,
                        bias_cur,
                        act_alpha,
                        act_beta,
                        act_type,
                        # TODO do we really need output scale to
                        # scale bias in kernel?
                        1.0 if output_scale is None else output_scale,  # output_scale
                        channel_scale,  # scale
                        add_input.features if add_input is not None else None,
                        output_add_scale,
                        output_dtype,
                    )

        if bias_for_training is not None:
            out_features += bias_for_training
        if input.benchmark:
            torch.cuda.synchronize()
            interval = time.time() - t
            out_tensor.benchmark_record[name]["time"].append(interval)
            out_tensor.benchmark_record[name]["num_points"].append(features.shape[0])
            out_tensor.benchmark_record[name]["num_out_points"].append(out_features.shape[0])
        if not self.subm and not self.inverse and self.record_voxel_count:
            if hasattr(self, _MAX_NUM_VOXELS_DURING_TRAINING):
                ops.maximum_value_int_(getattr(self, _MAX_NUM_VOXELS_DURING_TRAINING), outids.shape[0])
        out_tensor = out_tensor.replace_feature(out_features)
        out_tensor.indices = outids
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = out_spatial_shape
        if add_input is not None and not is_int8:
            # in int8, we apply add + act in kernel.
            out_tensor = out_tensor.replace_feature(
                _apply_act(out_tensor.features + add_input.features, self.act_type, self.act_alpha, self.act_beta)
            )

        return out_tensor


class SparseConv3d(SparseConvolution):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: Optional[ConvAlgo] = None,
        fp32_accum: Optional[bool] = None,
        record_voxel_count: bool = False,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SubMConv3d(SparseConvolution):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: Optional[ConvAlgo] = None,
        fp32_accum: Optional[bool] = None,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )
