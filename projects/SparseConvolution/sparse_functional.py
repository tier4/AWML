# Copyright 2019 Yan Yan
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
from typing import Any

from mmcv.utils import ext_loader
import sparse_ops as ops
import torch
from torch.autograd import Function

ext_module = ext_loader.load_ext(
    "_ext",
    [
        "get_indice_pairs_2d_forward",
        "get_indice_pairs_3d_forward",
        "get_indice_pairs_4d_forward",
        "get_indice_pairs_2d_backward",
        "get_indice_pairs_3d_backward",
    ],
)

from torch.onnx.symbolic_helper import _get_tensor_dim_size
from torch.onnx.symbolic_helper import _get_tensor_sizes


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):

    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[
            i
        ] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i]
        output_size.append(size)
    return output_size


def get_indice_pairs(
    indices,
    batch_size,
    spatial_shape,
    ksize=3,
    stride=1,
    padding=0,
    dilation=1,
    out_padding=0,
    subm=False,
    transpose=False,
    grid=None,
):
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    for d, s in zip(dilation, stride):
        assert any([s == 1, d == 1]), "don't support this."

    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(
                spatial_shape, ksize, stride, padding, dilation, out_padding
            )
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride, padding, dilation)

    else:
        out_shape = spatial_shape

    if grid is None:
        if ndim == 2:
            get_indice_pairs_func = get_indice_pairs_2d_forward
        elif ndim == 3:
            get_indice_pairs_func = get_indice_pairs_3d_forward
        elif ndim == 4:
            get_indice_pairs_func = get_indice_pairs_4d_forward
        else:
            raise NotImplementedError

        outids, indice_pairs, indice_pair_num, num_activate_out = get_indice_pairs_func(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        )

        return (
            outids,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
        )  # Note(kenzo): seems to be needed by the tracer
    else:
        if ndim == 2:
            get_indice_pairs_func = get_indice_pairs_2d_backward
        elif ndim == 3:
            get_indice_pairs_func = get_indice_pairs_3d_backward
        else:
            raise NotImplementedError
        outids, indice_pairs, indice_pair_num, num_activate_out = get_indice_pairs_func(
            indices,
            grid,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        )

        return (
            outids,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
        )  # Note(kenzo): seems to be needed by the tracer


class GetIndicePairs2dForward(Function):

    @staticmethod
    def symbolic(
        g,
        indices,
        batch_size,
        out_shape,
        spatial_shape,
        ksize,
        stride,
        padding,
        dilation,
        out_padding,
        subm,
        transpose,
    ):
        outputs = g.op(
            "mydomain::GetIndicePairs2dForward",
            indices,
            batch_size_i=batch_size,
            out_shape_i=out_shape,
            spatial_shape_i=spatial_shape,
            ksize_i=ksize,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            out_padding_i=out_padding,
            subm_i=subm,
            transpose_i=transpose,
            outputs=4,
        )

        indices_shape = _get_tensor_sizes(indices)
        if indices_shape is not None and hasattr(indices.type(), "with_sizes"):
            output_type_1 = indices.type().with_sizes([None, 3])

            output_type_2 = indices.type().with_sizes([None, 2, None])

            output_type_3 = indices.type().with_sizes([None])

            output_type_4 = indices.type().with_sizes([1])

            outputs[0].setType(output_type_1)
            outputs[1].setType(output_type_2)
            outputs[2].setType(output_type_3)
            outputs[3].setType(output_type_4)
        return outputs

    @staticmethod
    def forward(
        ctx,
        indices,
        batch_size,
        out_shape,
        spatial_shape,
        ksize,
        stride,
        padding,
        dilation,
        out_padding,
        subm,
        transpose,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm,
            transpose,
        )

        out_shape = out_shape.tolist() if isinstance(out_shape, torch.Tensor) else out_shape
        spatial_shape = (
            spatial_shape.tolist() if isinstance(spatial_shape, torch.Tensor) else spatial_shape
        )
        outids, indice_pairs, indice_pair_num = ext_module.get_indice_pairs_2d_forward(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm,
            transpose,
        )
        num_activate_out = outids.size(0)

        return outids, indice_pairs, indice_pair_num, num_activate_out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        return None, None, None, None, None, None, None, None, None, None


class GetIndicePairs3dForward(Function):

    @staticmethod
    def symbolic(
        g,
        indices,
        out_shape,
        batch_size,
        spatial_shape,
        ksize,
        stride,
        padding,
        dilation,
        out_padding,
        subm,
        transpose,
    ):
        outputs = g.op(
            "mydomain::GetIndicePairs3dForward",
            indices,
            batch_size_i=batch_size,
            out_shape_i=out_shape,
            spatial_shape_i=spatial_shape,
            ksize_i=ksize,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            out_padding_i=out_padding,
            subm_i=subm,
            transpose_i=transpose,
            outputs=4,
        )

        indices_shape = _get_tensor_sizes(indices)
        if indices_shape is not None and hasattr(indices.type(), "with_sizes"):
            output_type_1 = indices.type().with_sizes([None, 4])

            output_type_2 = indices.type().with_sizes([None, 2, None])

            output_type_3 = indices.type().with_sizes([None])

            output_type_4 = indices.type().with_sizes([1])

            outputs[0].setType(output_type_1)
            outputs[1].setType(output_type_2)
            outputs[2].setType(output_type_3)
            outputs[3].setType(output_type_4)
        return outputs

    @staticmethod
    def forward(
        ctx,
        indices,
        batch_size,
        out_shape,
        spatial_shape,
        ksize,
        stride,
        padding,
        dilation,
        out_padding,
        subm,
        transpose,
    ) -> torch.Tensor:

        ctx.save_for_backward(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm,
            transpose,
        )

        out_shape = out_shape.tolist() if isinstance(out_shape, torch.Tensor) else out_shape
        spatial_shape = (
            spatial_shape.tolist() if isinstance(spatial_shape, torch.Tensor) else spatial_shape
        )
        outids, indice_pairs, indice_pair_num = ext_module.get_indice_pairs_3d_forward(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm,
            transpose,
        )
        num_activate_out = outids.size(0)

        return outids, indice_pairs, indice_pair_num, num_activate_out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        return None, None, None, None, None, None, None, None, None, None


class GetIndicePairs4dForward(Function):

    @staticmethod
    def symbolic(
        g,
        indices,
        out_shape,
        batch_size,
        spatial_shape,
        ksize,
        stride,
        padding,
        dilation,
        out_padding,
        subm,
        transpose,
    ):
        outputs = g.op(
            "mydomain::GetIndicePairs4dForward",
            indices,
            batch_size_i=batch_size,
            out_shape_i=out_shape,
            spatial_shape_i=spatial_shape,
            ksize_i=ksize,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            out_padding_i=out_padding,
            subm_i=subm,
            transpose_i=transpose,
            outputs=4,
        )

        indices_shape = _get_tensor_sizes(indices)
        if indices_shape is not None and hasattr(indices.type(), "with_sizes"):
            output_type_1 = indices.type().with_sizes([None, 5])

            output_type_2 = indices.type().with_sizes([None, 2, None])

            output_type_3 = indices.type().with_sizes([None])

            output_type_4 = indices.type().with_sizes([1])

            outputs[0].setType(output_type_1)
            outputs[1].setType(output_type_2)
            outputs[2].setType(output_type_3)
            outputs[3].setType(output_type_4)
        return outputs

    @staticmethod
    def forward(
        ctx,
        indices,
        batch_size,
        out_shape,
        spatial_shape,
        ksize,
        stride,
        padding,
        dilation,
        out_padding,
        subm,
        transpose,
    ) -> torch.Tensor:

        ctx.save_for_backward(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm,
            transpose,
        )

        out_shape = out_shape.tolist() if isinstance(out_shape, torch.Tensor) else out_shape
        spatial_shape = (
            spatial_shape.tolist() if isinstance(spatial_shape, torch.Tensor) else spatial_shape
        )
        outids, indice_pairs, indice_pair_num = ext_module.get_indice_pairs_3d_forward(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm,
            transpose,
        )
        num_activate_out = outids.size(0)

        return outids, indice_pairs, indice_pair_num, num_activate_out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        return None, None, None, None, None, None, None, None, None, None


class SparseConvFunction(Function):
    """Sparse Convolution.

    Please refer to `SECOND <https://www.mdpi.com/1424-8220/18/10/3337>`_ for
    more details.
    """

    @staticmethod
    def symbolic(g, features, filters, indice_pairs, indice_pairs_num, num_activate_out):
        output = g.op(
            "mydomain::SparseConv",
            features,
            filters,
            indice_pairs,
            indice_pairs_num,
            num_activate_out,
            outputs=1,
        )

        # do shape inference and set it via setType
        features_shape = _get_tensor_sizes(features)
        if features_shape is not None and hasattr(features.type(), "with_sizes"):
            output_type = features.type().with_sizes(
                features_shape[0:1] + [_get_tensor_dim_size(filters, -1)]
            )

            output.setType(output_type)
        return output

    @staticmethod
    def forward(
        ctx: Any,
        features: torch.Tensor,
        filters: torch.nn.Parameter,
        indice_pairs: torch.Tensor,
        indice_pair_num: torch.Tensor,
        out_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        num_activate_out = out_ids.shape[0]
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        out_features = ops.indice_conv(
            features, filters, indice_pairs, indice_pair_num, num_activate_out, False
        )
        return out_features

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num, False
        )

        return input_bp, filters_bp, None, None, None


class SparseInverseConvFunction(Function):

    @staticmethod
    def symbolic(g, features, filters, indice_pairs, indice_pairs_num, num_activate_out):
        return g.op(
            "mydomain::SparseInverseConv",
            features,
            filters,
            indice_pairs,
            indice_pairs,
            indice_pairs_num,
            num_activate_out,
            outputs=1,
        )

    @staticmethod
    def forward(
        ctx: Any,
        features: torch.Tensor,
        filters: torch.nn.Parameter,
        indice_pairs: torch.Tensor,
        indice_pair_num: torch.Tensor,
        num_activate_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return ops.indice_conv(
            features, filters, indice_pairs, indice_pair_num, num_activate_out, True, False
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num, True, False
        )

        return input_bp, filters_bp, None, None, None


class SubMConvFunction(Function):

    @staticmethod
    def symbolic(g, features, filters, indice_pairs, indice_pairs_num, num_activate_out):
        return g.op(
            "mydomain::SubMConv",
            features,
            filters,
            indice_pairs,
            indice_pairs,
            indice_pairs_num,
            num_activate_out,
            outputs=1,
        )

    @staticmethod
    def forward(
        ctx: Any,
        features: torch.Tensor,
        filters: torch.nn.Parameter,
        indice_pairs: torch.Tensor,
        indice_pair_num: torch.Tensor,
        num_activate_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        return ops.indice_conv(
            features, filters, indice_pairs, indice_pair_num, num_activate_out, False, True
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        input_bp, filters_bp = ops.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num, False, True
        )

        return input_bp, filters_bp, None, None, None


class SparseMaxPoolFunction(Function):

    @staticmethod
    def symbolic(g, features, indice_pairs, indice_pairs_num, num_activate_out):
        return g.op(
            "mydomain::SparseMaxPool",
            features,
            indice_pairs,
            indice_pairs_num,
            num_activate_out,
            outputs=1,
        )

    @staticmethod
    def forward(
        ctx,
        features: torch.Tensor,
        indice_pairs: torch.Tensor,
        indice_pair_num: torch.Tensor,
        num_activate_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from sparse maxpooling.
        """
        out = ops.indice_maxpool(features, indice_pairs, indice_pair_num, num_activate_out)
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, out)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        indice_pairs, indice_pair_num, features, out = ctx.saved_tensors
        input_bp = ops.indice_maxpool_backward(
            features, out, grad_output, indice_pairs, indice_pair_num
        )
        return input_bp, None, None, None


get_indice_pairs_2d_forward = GetIndicePairs2dForward.apply
get_indice_pairs_3d_forward = GetIndicePairs3dForward.apply
get_indice_pairs_4d_forward = GetIndicePairs4dForward.apply

get_indice_pairs_2d_backward = ext_module.get_indice_pairs_2d_backward
get_indice_pairs_3d_backward = ext_module.get_indice_pairs_3d_backward

indice_conv = SparseConvFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_maxpool = SparseMaxPoolFunction.apply
