# Copyright (c) OpenMMLab. All rights reserved.

# Copied from
# https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/
# mmdeploy/codebase/mmdet3d/deploy.voxel_detection.py
# NOTE(knzo25): This patches the preprocessing step to perform
# the voxel reduction step OUTSIDE the graph

from typing import Dict, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmdeploy.codebase.mmdet3d import MMDetection3d
from mmdeploy.codebase.mmdet3d.deploy.voxel_detection import VoxelDetection as _VoxelDetection
from mmdeploy.utils import Task
from mmengine.dataset import pseudo_collate
from mmengine.model import BaseDataPreprocessor

MMDET3D_TASK = MMDetection3d.task_registry


@MMDET3D_TASK.register_module(Task.VOXEL_DETECTION.value, force=True)
class VoxelDetection(_VoxelDetection):

    def __init__(self, model_cfg: mmengine.Config, deploy_cfg: mmengine.Config, device: str):
        super().__init__(model_cfg, deploy_cfg, device)

    def create_input(
        self,
        batch: Union[str, Sequence[str]],
        data_preprocessor: Optional[BaseDataPreprocessor] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> Tuple[Dict, torch.Tensor]:

        data = [batch]
        collate_data = pseudo_collate(data)
        data[0]["inputs"]["points"] = data[0]["inputs"]["points"].to(self.device)

        """ cam2img = data[0]["data_samples"].cam2img
        cam2lidar = data[0]["data_samples"].cam2lidar
        lidar2image = data[0]["data_samples"].lidar2img
        lidar2camera = data[0]["data_samples"].lidar2cam
        img_aux_matrix = data[0]["data_samples"].img_aug_matrix

        import pickle
        d = {}
        d["cam2img"] = cam2img
        d["cam2lidar"] = cam2lidar
        d["lidar2image"] = lidar2image
        d["lidar2camera"] = lidar2camera
        d["img_aux_matrix"] = img_aux_matrix
        d["points"] = data[0]['inputs']['points'].cpu().numpy()

        with open("example.pkl", "wb") as f:
            pickle.dump(d, f) """

        assert data_preprocessor is not None
        collate_data = data_preprocessor(collate_data, False)
        points = collate_data["inputs"]["points"][0]
        voxels = collate_data["inputs"]["voxels"]
        inputs = [voxels["voxels"], voxels["num_points"], voxels["coors"]]

        feats = voxels["voxels"]
        num_points_per_voxel = voxels["num_points"]

        # NOTE(knzo25): preprocessing in BEVFusion and the
        # data_preprocessor work different.
        # The original code/model uses [batch, x, y, z]
        # but the data_preprocessor used here uses [batch, z, y, x]
        # In addition, the deployment's voxelization uses [z, y, x]
        # Since this is outside the graph we format it as [z, y, x]
        # and convert it to [batch, x, y, z] inside the graph
        coors = voxels["coors"]
        coors = coors[:, 1:]

        if "imgs" not in self.deploy_cfg.onnx_config.input_names:
            return collate_data, [feats, coors, num_points_per_voxel]

        # NOTE(knzo25): we want to load images from the camera
        # directly to the model in TensorRT
        img = batch["inputs"]["img"].type(torch.uint8)

        data_samples = collate_data["data_samples"][0]
        lidar2image = feats.new_tensor(data_samples.lidar2img)
        cam2image = feats.new_tensor(data_samples.cam2img)
        camera2lidar = feats.new_tensor(data_samples.cam2lidar)

        # NOTE(knzo25): ONNX/TensorRT do not support matrix inversion,
        # so they are taken out of the graph
        cam2image_inverse = torch.inverse(cam2image)

        # The extrinsics-related variables should only be computed once,
        # so we bring them outside the graph. Additionally, they require
        # argsort over the threshold available in TensorRT
        img_aux_matrix = feats.new_tensor(np.stack(collate_data["data_samples"][0].img_aug_matrix))
        img_aux_matrix_inverse = torch.inverse(img_aux_matrix)
        geom = model.view_transform.get_geometry(
            camera2lidar[..., :3, :3].unsqueeze(0).to(torch.device("cuda")),
            camera2lidar[..., :3, 3].unsqueeze(0).to(torch.device("cuda")),
            cam2image_inverse[..., :3, :3].unsqueeze(0).to(torch.device("cuda")),
            img_aux_matrix_inverse[..., :3, :3].unsqueeze(0).to(torch.device("cuda")),
            img_aux_matrix[..., :3, 3].unsqueeze(0).to(torch.device("cuda")),
        )

        geom_feats, kept, ranks, indices = model.view_transform.bev_pool_aux(geom)

        # TODO(knzo25): just a test. remove
        """ import pickle
        data = {}
        data["geom"] = geom.cpu()
        data["geom_feats"] = geom_feats.cpu()
        data["kept"] = kept.cpu()
        data["ranks"] = ranks.cpu()
        data["indices"] = indices.cpu()

        with open("precomputed_features.pkl", "wb") as f:
            pickle.dump(data, f) """

        inputs = [
            feats,
            coors,
            num_points_per_voxel,
            points,
            torch.ones((img.size(0)), device=img.device),
            img,
            lidar2image,
            # NOTE(knzo25): not used during export
            # but needed to comply with the signature
            cam2image,
            # NOTE(knzo25): not used during export
            # but needed to comply with the signature
            camera2lidar,
            geom_feats.int(),
            kept.bool(),  # TensorRT treats bool as uint8
            ranks,
            indices,
        ]

        return collate_data, inputs
