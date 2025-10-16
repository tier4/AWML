# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------

import mmcv
import numpy as np
import pyquaternion
import torch
from mmcv.transforms import Compose
from mmdet.registry import TRANSFORMS
from mmpretrain.registry import TRANSFORMS as TRANSFORMS_MMPRETRAIN
from PIL import Image


@TRANSFORMS.register_module()
class PadMultiViewImage:
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results["img"]]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val) for img in results["img"]
            ]
        # padded_img = results["img"]
        results["img_shape"] = [img.shape for img in results["img"]]
        results["img"] = padded_img
        results["img_metas"]["pad_shape"] = results["img"][0].shape
        results["img_metas"]["pad_fix_size"] = self.size
        results["img_metas"]["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = torch.stack(
            [
                torch.tensor(mmcv.imnormalize(img, self.mean, self.std, self.to_rgb).transpose(2, 0, 1))
                for img in results["img"]
            ]
        )
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@TRANSFORMS.register_module()
class ResizeCropFlipRotImage:
    """
    Applies image augmentation including resize, crop, flip, and rotation to multi-view images.

    This transform performs comprehensive image augmentation for multi-view camera data,
    including resizing, cropping, horizontal flipping, and rotation. It also handles
    the transformation of associated 2D bounding boxes, center points, labels, and depths
    when `with_2d=True`.

    The transform maintains consistency between image transformations and camera intrinsics
    by updating the intrinsic matrices accordingly.

    Args:
        data_aug_conf (dict): Configuration dictionary for data augmentation parameters.
            Contains the following keys:

            - **resize_lim** (float or tuple): Resize scale limits.
                - If float: Random scale sampled from [aspect_ratio - resize_lim, aspect_ratio + resize_lim]
                  where aspect_ratio = max(final_H/H, final_W/W)
                - If tuple (min, max): Random scale sampled from [min, max]

            - **final_dim** (tuple): Target output dimensions (height, width) after all transformations.

            - **bot_pct_lim** (tuple): Bottom percentage limits (min, max) for crop positioning.
                Controls how much of the bottom portion of the resized image to include.
                (0.0, 0.0) means crop from bottom, (1.0, 1.0) means crop from top.

            - **rot_lim** (tuple): Rotation angle limits in degrees (min, max).
                Currently only (0.0, 0.0) is supported (no rotation).

            - **rand_flip** (bool): Whether to apply random horizontal flipping.

        with_2d (bool, optional): Whether to transform 2D annotations (bboxes, centers, etc.).
            Default: True.

        filter_invisible (bool, optional): Whether to filter out invisible bounding boxes
            after transformation. Default: True.

        training (bool, optional): Whether in training mode. Affects augmentation behavior.
            In training mode, random augmentations are applied. In test mode, deterministic
            center crop is used. Default: True.

    Example:
        ```python
        # Configuration for training
        ida_aug_conf = {
            "resize_lim": (0.42, 0.46),     # Random scale the image between 0.42-0.46 to match final_dim
            "final_dim": (480, 640),        # Output size 480x640
            "bot_pct_lim": (0.0, 0.0),      # Crop from bottom
            "rot_lim": (0.0, 0.0),          # No rotation
            "rand_flip": True,              # Enable random flipping
        }

        # Configuration for testing
        ida_aug_conf_test = {
            "resize_lim": 0.02,             # Scale variation ±0.02 around aspect ratio
            "final_dim": (480, 640),        # Output size 480x640
            "bot_pct_lim": (0.0, 0.0),      # Crop from bottom
            "rot_lim": (0.0, 0.0),          # No rotation
            "rand_flip": False,             # No flipping in test
        }

        transform = ResizeCropFlipRotImage(
            data_aug_conf=ida_aug_conf,
            with_2d=True,
            filter_invisible=True,
            training=True
        )
        ```

    Note:
        - Rotation is currently not supported (rot_lim must be (0.0, 0.0))
        - The transform updates camera intrinsics to maintain geometric consistency
        - When filter_invisible=True, overlapping bounding boxes are filtered based on depth
        - The transform handles both single and multi-view image scenarios
    """

    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    def __call__(self, results):

        imgs = results["img"]
        H, W = imgs[0].shape[:2]
        N = len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers_2d = []
        new_gt_labels = []
        new_depths = []
        assert self.data_aug_conf["rot_lim"] == (0.0, 0.0), "Rotation is not currently supported"

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation(H, W)

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if self.with_2d:  # sync_2d bbox labels
                gt_bboxes = results["gt_bboxes"][i]
                centers_2d = results["centers_2d"][i]
                gt_labels = results["gt_bboxes_labels"][i]
                depths = results["depths"][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers_2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes,
                        centers_2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers_2d, gt_labels, depths = self._filter_invisible(
                        gt_bboxes, centers_2d, gt_labels, depths
                    )

                new_gt_bboxes.append(gt_bboxes)
                new_centers_2d.append(centers_2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            new_imgs.append(np.array(img).astype(np.float32))
            results["intrinsics"][i][:3, :3] = ida_mat @ results["intrinsics"][i]

        if self.with_2d:  # sync_2d bbox labels
            results["gt_bboxes"] = new_gt_bboxes
            results["centers_2d"] = new_centers_2d
            results["gt_bboxes_labels"] = new_gt_labels
            results["depths"] = new_depths
        results["img"] = new_imgs
        results["lidar2img"] = [
            np.concatenate([results["intrinsics"][i] @ results["extrinsics"][i][:3, :], np.array([[0, 0, 0, 1]])])
            for i in range(len(results["extrinsics"]))
        ]

        return results

    def _bboxes_transform(self, bboxes, centers_2d, gt_labels, depths, resize, crop, flip):
        assert len(bboxes) == len(centers_2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers_2d = centers_2d * resize
        centers_2d[:, 0] = centers_2d[:, 0] - crop[0]
        centers_2d[:, 1] = centers_2d[:, 1] - crop[1]
        centers_2d[:, 0] = np.clip(centers_2d[:, 0], 0, fW)
        centers_2d[:, 1] = np.clip(centers_2d[:, 1], 0, fH)
        if flip:
            centers_2d[:, 0] = fW - centers_2d[:, 0]

        centers_2d = centers_2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers_2d, gt_labels, depths

    def _filter_invisible(self, bboxes, centers_2d, gt_labels, depths):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers_2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH, fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind="stable")
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers_2d = centers_2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers_2d = centers_2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers_2d, gt_labels, depths

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self, H, W):
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            if isinstance(self.data_aug_conf["resize_lim"], (int, float)):
                aspect_ratio = max(fH / H, fW / W)
                resize = np.random.uniform(
                    aspect_ratio - self.data_aug_conf["resize_lim"], aspect_ratio + self.data_aug_conf["resize_lim"]
                )
            else:
                resize = np.random.uniform(*self.data_aug_conf["resize_lim"])

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@TRANSFORMS.register_module()
class GlobalRotScaleTransImage:
    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        # random rotate
        translation_std = np.array(self.translation_std, dtype=np.float32)

        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        trans = np.random.normal(scale=translation_std, size=3).T

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1
        results["gt_bboxes_3d"].rotate(np.array(rot_angle))

        # random scale
        self._scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # random translate
        self._trans_xyz(results, trans)
        results["gt_bboxes_3d"].translate(trans)

        return results

    def _trans_xyz(self, results, trans):
        trans_mat = torch.eye(4, 4)
        trans_mat[:3, -1] = torch.from_numpy(trans).reshape(1, 3)
        trans_mat_inv = torch.inverse(trans_mat)
        num_view = len(results["lidar2img"])
        results["ego_pose"] = (torch.tensor(results["ego_pose"]).float() @ trans_mat_inv).numpy()
        results["ego_pose_inv"] = (trans_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()

        for view in range(num_view):
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ trans_mat_inv).numpy()
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ trans_mat_inv).numpy()

    def _rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, rot_sin, 0, 0], [-rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        results["ego_pose"] = (torch.tensor(results["ego_pose"]).float() @ rot_mat_inv).numpy()
        results["ego_pose_inv"] = (rot_mat.float() @ torch.tensor(results["ego_pose_inv"]).float()).numpy()
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()

    def _scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = torch.inverse(scale_mat)

        results["ego_pose"] = (torch.tensor(results["ego_pose"]).float() @ scale_mat_inv).numpy()
        results["ego_pose_inv"] = (scale_mat @ torch.tensor(results["ego_pose_inv"]).float()).numpy()

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ scale_mat_inv).numpy()
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ scale_mat_inv).numpy()


@TRANSFORMS.register_module()
class ConvertTo3dGlobal:

    def __call__(self, results):

        box = results["gt_bboxes_3d"]
        box.tensor[:, :3] = box.gravity_center
        l2e_matrix = results["l2e_matrix"]
        e2g_matrix = results["e2g_matrix"]

        box.rotate(l2e_matrix[:3, :3])
        box.translate(l2e_matrix[:3, 3])
        # filter det in ego.

        box.rotate(e2g_matrix[:3, :3])
        box.translate(e2g_matrix[:3, 3])

        results["gt_bboxes_3d"] = box

        return results


@TRANSFORMS.register_module()
class Filter2DByRange:
    def __init__(self, range_2d: float = 61.2):
        self.range_2d = range_2d

    def __call__(self, results):
        N = len(results["img"])
        for i in range(N):
            mask = results["depths"][i] < self.range_2d
            results["centers_2d"][i] = results["centers_2d"][i][mask]
            results["gt_bboxes_labels"][i] = results["gt_bboxes_labels"][i][mask]
            results["gt_bboxes"][i] = results["gt_bboxes"][i][mask]
            results["depths"][i] = results["depths"][i][mask]
        return results


@TRANSFORMS.register_module()
class ImageAugmentation:
    """
    Applies a series of image augmentations with a given probability.

    This class wraps a sequence of transformations defined using mmcv's
    `Compose` and applies them to the image(s) in the input `results` dictionary
    with a probability `p`.  It's designed to integrate with the mmpretrain
    framework.

    Args:
        transforms (list): A list of transformation dictionaries, where each
            dictionary defines a transformation to be applied.  These
            dictionaries should be compatible with `mmpretrain.registry.TRANSFORMS.build`.
            For example:
            ```
            transforms = [
                dict(type='RandomResizedCrop', size=224),
                dict(type='RandomFlip', flip_prob=0.5),
            ]
            ```
        p (float, optional): The probability with which the augmentations are
            applied. Defaults to 0.75.

    Returns:
        dict: The input `results` dictionary with the image(s) potentially
            augmented.  The image(s) are accessed via `results["img"]`.  If
            multiple images are present (e.g., in multi-view testing), `results["img"]`
            should be a list of images.

    Example:
        ```python
        transforms = [
            dict(type='RandomResizedCrop', size=224),
            dict(type='RandomFlip', flip_prob=0.5)
        ]
        augmentation = ImageAugmentation(transforms, p=0.5)

        # Example usage with a single image:
        results = {'img': np.random.randint(0, 256, size=(256, 256, 3)).astype(np.uint8)}
        augmented_results = augmentation(results)
        print(augmented_results['img'].shape) # Output shape after transformation

        # Example usage with multiple images (e.g., multi-view):
        results_multi = {'img': [
            np.random.randint(0, 256, size=(256, 256, 3)).astype(np.uint8),
            np.random.randint(0, 256, size=(256, 256, 3)).astype(np.uint8)
        ]}
        augmented_results_multi = augmentation(results_multi)
        for img in augmented_results_multi['img']:
          print(img.shape) # Output shape after transformation for each image

        ```
    """

    def __init__(self, transforms: [], p=0.75):
        self.transforms = Compose([TRANSFORMS_MMPRETRAIN.build(t) for t in transforms])
        self.p = p

    def __call__(self, results):
        if self.transforms:
            for i, image in enumerate(results["img"]):
                if np.random.rand() < self.p:
                    results["img"][i] = self.transforms({"img": image.astype(np.uint8)})["img"]
        return results
