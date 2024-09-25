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

from mmcv.utils import ext_loader
import torch

ext_module = ext_loader.load_ext(
    "_ext",
    [
        "get_indice_pairs_2d_forward",
        "get_indice_pairs_3d_forward",
        "get_indice_pairs_4d_forward",
        "get_indice_pairs_2d_backward",
        "get_indice_pairs_3d_backward",
        "indice_conv_forward",
        "indice_conv_backward",
        "fused_indice_conv_forward",
        "indice_maxpool_forward",
        "indice_maxpool_backward",
    ],
)


def indice_conv(
    features, filters, indice_pairs, indice_pair_num, num_activate_out, inverse=False, subm=False
):
    if filters.dtype == torch.float32 or filters.dtype == torch.half:
        return ext_module.indice_conv_forward(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            int(inverse),
            int(subm),
        )
    else:
        raise NotImplementedError


def fused_indice_conv(
    features, filters, bias, indice_pairs, indice_pair_num, num_activate_out, inverse, subm
):
    if features.dtype == torch.half or filters.dtypes == torch.float32:
        func = ext_module.fused_indice_conv_forward
    else:
        raise NotImplementedError

    return func(
        features,
        filters,
        bias,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        int(inverse),
        int(subm),
    )


def indice_conv_backward(
    features, filters, out_bp, indice_pairs, indice_pair_num, inverse=False, subm=False
):
    if filters.dtype == torch.float32 or filters.dtype == torch.half:
        return ext_module.indice_conv_backward(
            features, filters, out_bp, indice_pairs, indice_pair_num, int(inverse), int(subm)
        )
    else:
        raise NotImplementedError


def indice_maxpool(features, indice_pairs, indice_pair_num, num_activate_out):
    if features.dtype == torch.float32 or features.dtype == torch.half:
        return ext_module.indice_maxpool_forward(
            features, indice_pairs, indice_pair_num, num_activate_out
        )
    else:
        raise NotImplementedError


def indice_maxpool_backward(features, out_features, out_bp, indice_pairs, indice_pair_num):
    if features.dtype == torch.float32 or features.dtype == torch.half:
        return ext_module.indice_maxpool_backward(
            features, out_features, out_bp, indice_pairs, indice_pair_num
        )
    else:
        raise NotImplementedError
