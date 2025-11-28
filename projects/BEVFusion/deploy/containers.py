import torch
import torch.nn.functional as F

# Wrapper Classes for onnx conversion


class TrtBevFusionImageBackboneContainer(torch.nn.Module):
    def __init__(self, mod, mean, std) -> None:
        super().__init__()
        self.mod = mod
        self.images_mean = mean
        self.images_std = std

    def forward(
        self,
        imgs,
        points,
        lidar2image,
        cam2image,
        cam2image_inverse,
        camera2lidar,
        img_aug_matrix,
        img_aug_matrix_inverse,
        lidar_aug_matrix,
        lidar_aug_matrix_inverse,
        geom_feats,
        kept,
        ranks,
        indices,
    ):

        mod = self.mod
        imgs = (imgs - self.images_mean) / self.images_std

        return mod.extract_img_feat(
            imgs,
            points,
            lidar2image,
            cam2image,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas=None,
            img_aug_matrix_inverse=img_aug_matrix_inverse,
            camera_intrinsics_inverse=cam2image_inverse,
            lidar_aug_matrix_inverse=lidar_aug_matrix_inverse,
            geom_feats=(geom_feats, kept, ranks, indices),
        )


class TrtBevFusionMainContainer(torch.nn.Module):
    def __init__(self, mod, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod = mod

    def forward(self, voxels, coors, num_points_per_voxel, image_feats=None):
        mod = self.mod

        features = []

        if image_feats is not None:
            features.append(image_feats)

        if coors.shape[1] == 3:
            num_points = coors.shape[0]
            coors = coors.flip(dims=[-1]).contiguous()  # [x, y, z]
            batch_coors = torch.zeros(num_points, 1).to(coors.device)
            coors = torch.cat([batch_coors, coors], dim=1).contiguous()

        pts_feature = mod.extract_pts_feat(voxels, coors, num_points_per_voxel)
        features.append(pts_feature)

        if mod.fusion_layer is not None:
            x = mod.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]
        x = mod.pts_backbone(x)
        x = mod.pts_neck(x)

        outputs = mod.bbox_head(x, None)[0][0]

        score = outputs["heatmap"].sigmoid()
        one_hot = F.one_hot(outputs["query_labels"], num_classes=score.size(1)).permute(0, 2, 1)
        score = score * outputs["query_heatmap_score"] * one_hot
        score = score[0].max(dim=0)[0]

        bbox_pred = torch.cat(
            [outputs["center"][0], outputs["height"][0], outputs["dim"][0], outputs["rot"][0], outputs["vel"][0]],
            dim=0,
        )
        return bbox_pred, score, outputs["query_labels"][0]
