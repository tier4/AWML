# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import cv2
from typing import List, Optional

import mmcv
from mmcv.transforms import BaseTransform
import numpy as np
import torch 

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from mmengine.fileio import get
from mmengine.logging import print_log
from PIL import Image 
import uuid
import matplotlib.pyplot as plt


def project_to_image(points, lidar2cam, cam2img, img_aug_matrix, lidar_aug_matrix):
    """Transform points from LiDAR to image coordinates."""
    # Undo lidar_aug
    points -= lidar_aug_matrix[:3, 3]
    points = (np.linalg.inv(lidar_aug_matrix[:3, :3]) @ points.T).T

    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = np.dot(lidar2cam, points_hom.T).T

    # Filter points behind the camera
    valid_mask = points_cam[:, 2] > 0

    points_cam_hom = np.hstack((points_cam[:, :3], np.ones((points_cam.shape[0], 1))))
    points_img = np.dot(cam2img, points_cam_hom.T).T
    points_img[:, :2] /= points_img[:, 2:3]

    # imgaug
    points_img = (img_aug_matrix[:3, :3] @ points_img[:, :3].T).T
    points_img += img_aug_matrix[:3, 3]

    return points_img[:, :2], valid_mask


def compute_bbox_and_centers(lidar2cam, cam2img, img_aug_matrix, lidar_aug_matrix, bboxes, labels, img_shape):
    """
    Compute the 2D bounding box, 3D center of the projected bounding box, and 3D center in LiDAR coordinates.

    Args:
        data_dict (dict): Contains the image path, lidar2cam, and cam2img transformation matrices.
        bboxes (object): Contains the 3D bounding box corners and labels.
        labels (np.ndarray): Array of labels for each bbox
        img_shape (tuple): Image dimensions (H, W)

    Returns:
        tuple: Contains:
            - bboxes_2d: np.ndarray of shape (N, 4) for [x1, y1, x2, y2]
            - projected_centers: np.ndarray of shape (N, 2) for projected 3D centers
            - centers_3d: np.ndarray of shape (N, 3) for 3D centers in LiDAR coords
            - valid_labels: np.ndarray of shape (N,) containing labels for valid boxes
    """

    H, W, C = img_shape
    # Initialize lists to store valid results
    valid_bboxes_2d = []
    valid_projected_centers = []
    valid_image_depth = []
    valid_labels_list = []

    # cam2img = img_aug_matrix @ cam2img 

    # Loop through each bounding box
    for bbox_std, bbox, label in zip(bboxes, bboxes.corners, labels):
        # Project corners to image
        center_3d_lidar = bbox_std[:3].numpy()
        corners_img, valid_mask = project_to_image(
            np.concatenate([bbox, bbox.mean(0).reshape(1, 3)]), lidar2cam, cam2img, img_aug_matrix, lidar_aug_matrix
        )
        projected_center = corners_img[-1]

        corners_img = corners_img[:-1][valid_mask[:-1]]

        if len(corners_img) == 0:  # Skip if no corners are visible
            continue

        # Compute 2D bbox
        x_min, y_min = np.min(corners_img, axis=0)
        x_max, y_max = np.max(corners_img, axis=0)

        # Clip to image boundaries
        x_min = np.clip(x_min, 0, W)
        x_max = np.clip(x_max, 0, W)
        y_min = np.clip(y_min, 0, H)
        y_max = np.clip(y_max, 0, H)

        x_center = np.clip(projected_center[0], 0, W)
        y_center = np.clip(projected_center[1], 0, H)
        if x_min == x_max or y_min == y_max:
            continue

        valid_bboxes_2d.append([x_min, y_min, x_max, y_max])
        valid_projected_centers.append([x_center, y_center])
        valid_image_depth.append(np.sqrt((center_3d_lidar**2).sum()))
        valid_labels_list.append(label)

    if valid_bboxes_2d:
        bboxes_2d = np.array(valid_bboxes_2d)
        projected_centers = np.array(valid_projected_centers)
        object_depth = np.array(valid_image_depth)
        valid_labels = np.array(valid_labels_list)
    else:
        # Return empty arrays with correct shapes if no valid boxes
        bboxes_2d = np.zeros((0, 4))
        projected_centers = np.zeros((0, 2))
        object_depth = np.zeros((0,))
        valid_labels = np.zeros(0, dtype=int)

    return bboxes_2d, projected_centers, object_depth, valid_labels


def check_bbox_visibility_in_image(lidar2cam, cam2img, img_aug_matrix, bboxes, labels, img_shape, visibility=0.1):
    """
    Projects 3D bounding boxes into the image plane and determines visibility.

    Args:
        lidar2cam (np.ndarray): 4x4 transformation matrix from LiDAR to camera coordinates.
        cam2img (np.ndarray): 3x3 camera intrinsic matrix.
        bboxes (list): List of 3D bounding boxes. Each must have `.corners` attribute and be indexable.
        labels (list): List of labels corresponding to the bounding boxes.
        img_shape (tuple): Shape of the image in (Channels, Height, Width) format.
        visibility (float, optional): Minimum fraction (0–1) of projected 2D bbox area that must lie
            within the image to consider it visible. Defaults to 0.1.

    Returns:
        list: A list of booleans indicating if each bounding box is sufficiently visible.
    """
    C, H, W = img_shape
    is_visible = []
    # cam2img = img_aug_matrix @ cam2img

    for bbox_std, bbox, label in zip(bboxes, [b.corners for b in bboxes], labels):
        # Project corners + center to image space
        all_points = np.concatenate([bbox, bbox.mean(0).reshape(1, 3)], axis=0)
        corners_img, valid_mask = project_to_image(all_points, lidar2cam, cam2img)
        projected_center = corners_img[-1]
        corners_img = corners_img[:-1][valid_mask[:-1]]

        if len(corners_img) == 0:
            is_visible.append(False)
            continue

        # Compute full 2D bbox from all projected corners
        x_min, y_min = np.min(corners_img, axis=0)
        x_max, y_max = np.max(corners_img, axis=0)
        full_area = max(x_max - x_min, 0) * max(y_max - y_min, 0)

        if full_area == 0:
            is_visible.append(False)
            continue

        # Compute clipped bbox (intersection with image frame)
        x_min_clip = np.clip(x_min, 0, W)
        x_max_clip = np.clip(x_max, 0, W)
        y_min_clip = np.clip(y_min, 0, H)
        y_max_clip = np.clip(y_max, 0, H)
        visible_area = max(x_max_clip - x_min_clip, 0) * max(y_max_clip - y_min_clip, 0)

        visible_ratio = visible_area / full_area

        is_visible.append(visible_ratio >= visibility)

    return is_visible


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def __init__(
        self,
        camera_order: List[str],
        to_float32: bool = False,
        color_type: str = "unchanged",
        backend_args: Optional[dict] = None,
        num_views: int = 5,
        num_ref_frames: int = -1,
        test_mode: bool = False,
        set_default_scale: bool = True,
    ) -> None:
        self.camera_order = camera_order
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args
        self.num_views = num_views
        # num_ref_frames is used for multi-sweep loading
        self.num_ref_frames = num_ref_frames
        # when test_mode=False, we randomly select previous frames
        # otherwise, select the earliest one
        self.test_mode = test_mode
        self.set_default_scale = set_default_scale
        self.before_camera_info = dict()

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results["img_filename"]) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames, num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(num_frames, self.num_ref_frames - num_frames, replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results["img_filename"][choice * self.num_views : (choice + 1) * self.num_views]
            results["img_filename"] = select_filename
            for key in ["cam2img", "lidar2cam"]:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice * self.num_views : (choice + 1) * self.num_views]
                    results[key] = select_results
            for key in ["ego2global"]:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key](choice)]
                    results[key] = select_results
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ["lidar2cam"]:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results["ego2global"][choice_idx]
                        pad_prev_ego2global[: prev_ego2global.shape[0], : prev_ego2global.shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results["ego2global"][0]
                        pad_cur_ego2global[: cur_ego2global.shape[0], : cur_ego2global.shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views, (choice_idx + 1) * self.num_views):
                            results[key][result_idx] = results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], [], []

        # to fill None data
        # for _ , cam_item in results['images'].items():
        for camera_type in self.camera_order:
            if camera_type not in results["images"]:
                continue
            
            cam_item = results["images"][camera_type]
            # TODO (KokSeang): This sometime causes an error when we set num_workers > 1 during training,
            # it's likely due to multiprocessing in CPU. We should probably process this part when creating info files
            if cam_item["img_path"] is None:
                # print_log(f"Warning: None data for cam: {camera_type} in {results['images']}")
                # continue 
                cam_item = self.before_camera_info[camera_type]
                print_log("Warning: fill None data")
            else:
                self.before_camera_info[camera_type] = cam_item

            filename.append(cam_item["img_path"])
            lidar2cam.append(cam_item["lidar2cam"])

            lidar2cam_array = np.array(cam_item["lidar2cam"]).astype(np.float32)
            lidar2cam_rot = lidar2cam_array[:3, :3]
            lidar2cam_trans = lidar2cam_array[:3, 3:4]
            camera2lidar = np.eye(4)
            camera2lidar[:3, :3] = lidar2cam_rot.T
            camera2lidar[:3, 3:4] = -1 * np.matmul(lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))
            cam2lidar.append(camera2lidar)

            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item["cam2img"]).astype(np.float32)
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results["img_path"] = filename
        results["cam2img"] = np.stack(cam2img, axis=0)
        results["lidar2cam"] = np.stack(lidar2cam, axis=0)
        results["cam2lidar"] = np.stack(cam2lidar, axis=0)
        results["lidar2img"] = np.stack(lidar2img, axis=0)

        results["ori_cam2img"] = copy.deepcopy(results["cam2img"])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [get(name, backend_args=self.backend_args) for name in filename]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type, backend="pillow", channel_order="rgb")
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs]
        img = np.stack(imgs, axis=-1)
        # print(f"image_shape: {img.shape}")
        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape[:2]
        if self.set_default_scale:
            results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32), std=np.ones(num_channels, dtype=np.float32), to_rgb=False
        )
        results["num_views"] = self.num_views
        results["num_ref_frames"] = self.num_ref_frames
        return results


@TRANSFORMS.register_module()
class BEVFusionLoadAnnotations2D(BaseTransform):

    def transform(self, results):

        all_bboxes_2d, all_centers_2d, all_depths, all_labels = [], [], [], []
        vis_images = []
        lidar_aug_matrix = results.get("lidar_aug_matrix", np.eye(4))

        for i, k in enumerate(results["img"]):
            bboxes_2d, projected_centers, depths, valid_labels = compute_bbox_and_centers(
                results["lidar2cam"][i],
                results["cam2img"][i],
                results["img_aug_matrix"][i],
                lidar_aug_matrix,
                results["gt_bboxes_3d"],
                results["gt_labels_3d"],
                results["img"][i].shape,
            )
            # print(f"bboxes: {bboxes_2d.shape}, centers: {projected_centers.shape}, depths: {depths.shape}, labels: {valid_labels.shape}")
            all_bboxes_2d.append(bboxes_2d)
            all_centers_2d.append(projected_centers)
            all_depths.append(depths)
            all_labels.append(valid_labels)

            # ------------------------------
            # 🔵  Draw bounding boxes on image
            # ------------------------------
            # img = results["img"][i]
            # img_draw = img.copy()
            # for box, center, depth, label in zip(bboxes_2d, projected_centers, depths, valid_labels):
            
            #     # box = [x1, y1, x2, y2]
            #     x1, y1, x2, y2 = map(int, box)

            #     # draw bbox rectangle
            #     cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #     # draw center point
            #     cx, cy = map(int, center)
            #     cv2.circle(img_draw, (cx, cy), 3, (0, 0, 255), -1)

            #     # write depth and class label
            #     text = f"{label}, {depth:.1f}m"
            #     cv2.putText(
            #         img_draw, text, (x1, max(0, y1 - 5)),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            #     )
            # img_draw = Image.fromarray(img_draw.astype("uint8"), mode="RGB")
            # vis_images.append(img_draw)

        
        # Visualize image 
        # -----------------------------
        # 🔵 Save as subplot with 5 images
        # -----------------------------
        # num_images = len(vis_images)
        # cols = 5
        # rows = int(np.ceil(num_images / cols))

        # fig = plt.figure(figsize=(20, 4 * rows))

        # for idx, img in enumerate(vis_images):
        #     ax = fig.add_subplot(rows, cols, idx + 1)
        #     ax.imshow(img)
        #     ax.axis("off")
        #     ax.set_title(f"Camera {idx}")

        # fig.tight_layout()

        # Save figure to memory (as ndarray) or file
        # fig_path = f"work_dirs/bevfusion_image_2d_debug/2/debug_vis_{uuid.uuid4().hex}.png"
        # fig.savefig(fig_path, dpi=150)
        # plt.close(fig)

        results["depths"] = all_depths
        results["centers_2d"] = all_centers_2d
        results["gt_bboxes"] = all_bboxes_2d
        results["gt_bboxes_labels"] = all_labels
        return results


@TRANSFORMS.register_module()
class Filter3DBoxesinBlindSpot(BaseTransform):

    def __init__(self, visibility=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visibility = visibility

    def transform(self, results):
        visibility_mask = []
        lidar_aug_matrix = results.get("lidar_aug_matrix", np.eye(4))
        for i, k in enumerate(results["img"]):
            is_visible = check_bbox_visibility_in_image(
                results["lidar2cam"][i],
                results["cam2img"][i],
                results["img_aug_matrix"][i],
                lidar_aug_matrix,
                results["gt_bboxes_3d"],
                results["gt_labels_3d"],
                results["img"][i].shape,
            )
            visibility_mask.append(is_visible)
        visibility_mask = np.stack(visibility_mask).mean(0)

        return results
