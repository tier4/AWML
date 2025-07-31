import torch

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

    def forward(self, img):
        mod = self.mod
        return mod.extract_img_feat(img, 1).squeeze(1)
