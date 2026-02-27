import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Wrapper Classes for onnx conversion


class TrtBevFusionImageBackboneContainer(torch.nn.Module):
    def __init__(self, mod, mean, std) -> None:
        super().__init__()
        self.mod = mod
        self.images_mean = mean
        self.images_std = std

    def forward(self, imgs):

        mod = self.mod
        imgs = (imgs.float().unsqueeze(0) - self.images_mean) / self.images_std

        # No lidar augmentations expected during inference.
        return mod.get_image_backbone_features(imgs)[0]


class TrtBevFusionMainContainer(torch.nn.Module):

    def __init__(self, mod, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod = mod

    def forward(
        self,
        voxels,
        coors,
        num_points_per_voxel,
        points=None,
        lidar2img=None,
        img_aug_matrix=None,
        geom_feats=None,
        kept=None,
        ranks=None,
        indices=None,
        image_feats=None,
    ):
        mod = self.mod
        if coors.shape[1] == 3:
            num_points = coors.shape[0]
            coors = coors.flip(dims=[-1]).contiguous()  # [x, y, z]
            batch_coors = torch.zeros(num_points, 1).to(coors.device)
            coors = torch.cat([batch_coors, coors], dim=1).contiguous()

        batch_inputs_dict = {
            "voxels": {"voxels": voxels, "coors": coors, "num_points_per_voxel": num_points_per_voxel},
        }

        if points is not None:
            batch_inputs_dict["points"] = [points]

        if image_feats is not None:

            lidar_aug_matrix = torch.eye(4).unsqueeze(0).to(image_feats.device)

            batch_inputs_dict.update(
                {
                    "imgs": image_feats.unsqueeze(0),
                    "lidar2img": lidar2img.unsqueeze(0),
                    "cam2img": None,
                    "cam2lidar": None,
                    "img_aug_matrix": img_aug_matrix.unsqueeze(0),
                    "img_aug_matrix_inverse": None,
                    "lidar_aug_matrix": lidar_aug_matrix,
                    "lidar_aug_matrix_inverse": lidar_aug_matrix,
                    "geom_feats": (geom_feats, kept, ranks, indices),
                }
            )

        outputs = mod._forward(batch_inputs_dict, using_image_features=True)
        bbox_pred, score, label_pred = self.postprocessing(outputs)
        return bbox_pred, score, label_pred

    def postprocessing(self, outputs: dict):
        """Postprocess the outputs of the model to get the final predictions.

        Args:
            outputs (dict): The outputs of the model.

        Returns:
            dict: The final predictions.
        """
        # The following code is taken from
        # projects/BEVFusion/bevfusion/bevfusion_head.py
        # It is used to simplify the post process in deployment
        score = outputs["heatmap"].sigmoid()
        one_hot = F.one_hot(outputs["query_labels"], num_classes=score.size(1)).permute(0, 2, 1)
        score = score * outputs["query_heatmap_score"] * one_hot
        score = score[0].max(dim=0)[0]

        bbox_pred = torch.cat(
            [outputs["center"][0], outputs["height"][0], outputs["dim"][0], outputs["rot"][0], outputs["vel"][0]],
            dim=0,
        )

        return bbox_pred, score, outputs["query_labels"][0]


class TrtBevFusionCameraOnlyContainer(TrtBevFusionMainContainer):

    def __init__(self, mod, *args, **kwargs) -> None:
        super().__init__(mod=mod, *args, **kwargs)

    def forward(
        self,
        geom_feats,
        kept,
        ranks,
        indices,
        image_feats,
        points=None,
        lidar2img=None,
        img_aug_matrix=None,
    ):
        mod = self.mod
        lidar_aug_matrix = torch.eye(4).unsqueeze(0).to(image_feats.device)
        batch_inputs_dict = {
            "imgs": image_feats.unsqueeze(0),
            "cam2img": None,
            "cam2lidar": None,
            "lidar_aug_matrix": lidar_aug_matrix,
            "lidar_aug_matrix_inverse": lidar_aug_matrix,
            "geom_feats": (geom_feats, kept, ranks, indices),
            "points": [points] if points is not None else None,
            "lidar2img": lidar2img.unsqueeze(0) if lidar2img is not None else None,
            "img_aug_matrix": img_aug_matrix.unsqueeze(0) if img_aug_matrix is not None else None,
            "img_aug_matrix_inverse": None,
        }
        outputs = mod._forward(batch_inputs_dict, using_image_features=True)
        bbox_pred, score, label_pred = self.postprocessing(outputs)

        return bbox_pred, score, label_pred
