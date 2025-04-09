# Copyright (c) OpenMMLab. All rights reserved.

# Copied from
# https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/
# mmdeploy/codebase/mmdet3d/models/base.py
# NOTE(knzo25): This patches the forward method to
# control the inputs and outputs to the network in deployment

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter("mmdet3d.models.detectors.Base3DDetector.forward")  # noqa: E501
def basedetector__forward(
    self,
    voxels: Optional[torch.Tensor] = None,
    coors: Optional[torch.Tensor] = None,
    num_points_per_voxel: Optional[torch.Tensor] = None,
    points: Optional[torch.Tensor] = None,
    camera_mask: Optional[torch.Tensor] = None,
    imgs: Optional[torch.Tensor] = None,
    lidar2image: Optional[torch.Tensor] = None,
    # NOTE(knzo25): not used during export
    # but needed to comply with the signature
    cam2image: Optional[torch.Tensor] = None,
    # NOTE(knzo25): not used during export
    # but needed to comply with the signature
    camera2lidar: Optional[torch.Tensor] = None,
    geom_feats: Optional[torch.Tensor] = None,
    kept: Optional[torch.Tensor] = None,
    ranks: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    data_samples=None,
    **kwargs
) -> Tuple[List[torch.Tensor]]:

    # coors are generated in zyx order in the deployment's preprocessing
    # so we need to convert them to xyz order and add a batch dimension
    coors = coors[:, [2, 1, 0]]
    coors = torch.cat((torch.zeros(coors.size(0), 1, dtype=coors.dtype, device=coors.device), coors), dim=1)

    batch_inputs_dict = {
        "points": [points],
        "voxels": {"voxels": voxels, "coors": coors, "num_points_per_voxel": num_points_per_voxel},
    }

    if imgs is not None:

        # NOTE(knzo25): moved image normalization to the graph
        images_mean = kwargs["data_preprocessor"].mean.to(imgs.device)
        images_std = kwargs["data_preprocessor"].std.to(imgs.device)
        imgs = (imgs.float() - images_mean) / images_std

        # This is actually not used since we also use geom_feats as an input
        # However, it is needed to comply with signatures
        img_aug_matrix = imgs.new_tensor(np.stack(data_samples[0].img_aug_matrix))
        img_aug_matrix_inverse = imgs.new_tensor(np.stack([np.linalg.inv(x) for x in data_samples[0].img_aug_matrix]))
        lidar_aug_matrix = torch.eye(4).to(imgs.device)

        batch_inputs_dict.update(
            {
                "imgs": imgs.unsqueeze(dim=0),
                "lidar2img": lidar2image.unsqueeze(dim=0),
                "cam2img": cam2image.unsqueeze(dim=0),
                "cam2lidar": camera2lidar.unsqueeze(dim=0),
                "img_aug_matrix": img_aug_matrix.unsqueeze(dim=0),
                "img_aug_matrix_inverse": img_aug_matrix_inverse.unsqueeze(dim=0),
                "lidar_aug_matrix": lidar_aug_matrix.unsqueeze(dim=0),
                "lidar_aug_matrix_inverse": lidar_aug_matrix.unsqueeze(dim=0),
                "geom_feats": (geom_feats, kept, ranks, indices, camera_mask),
            }
        )

    outputs = self._forward(batch_inputs_dict, data_samples, **kwargs)

    # The following code is taken from
    # projects/BEVFusion/bevfusion/bevfusion_head.py
    # It is used to simplify the post process in deployment
    score = outputs["heatmap"].sigmoid()
    one_hot = F.one_hot(outputs["query_labels"], num_classes=score.size(1)).permute(0, 2, 1)
    score = score * outputs["query_heatmap_score"] * one_hot
    score = score[0].max(dim=0)[0]

    bbox_pred = torch.cat(
        [outputs["center"][0], outputs["height"][0], outputs["dim"][0], outputs["rot"][0], outputs["vel"][0]], dim=0
    )

    return bbox_pred, score, outputs["query_labels"][0]
