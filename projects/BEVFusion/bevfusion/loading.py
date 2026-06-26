# Copyright (c) OpenMMLab. All rights reserved.
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from mmengine.fileio import get
from mmengine.logging import print_log


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
        camera_orders: Dict[str, List[str]],
        to_float32: bool = False,
        color_type: str = "unchanged",
        backend_args: Optional[dict] = None,
        num_views: int = 5,
        num_ref_frames: int = -1,
        test_mode: bool = False,
        set_default_scale: bool = True,
    ) -> None:
        self.camera_orders = camera_orders
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
				self.camera_order_types = list(camera_orders.keys())

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
        vehicle_type = results.get("vehicle_type", None)
        if vehicle_type is None:
          camera_order = self.camera_orders[self.camera_order_types[0]]
        else:
          camera_order = self.camera_orders[vehicle_type]

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
        for camera_type in camera_order:
            if camera_type not in results["images"]:
                continue

            cam_item = results["images"][camera_type]
            # TODO (KokSeang): This sometime causes an error when we set num_workers > 1 during training,
            # it's likely due to multiprocessing in CPU. We should probably process this part when creating info files
            if cam_item["img_path"] is None:
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

        # Height, width, channels, num_views
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
class PointsToMultiViewImageDepths(BaseTransform):
    """Convert points to multi-view image depths.

    Args:
        points (np.ndarray): Points in the world coordinate system.
        img_shape (tuple): Shape of the image.
        cam2img (np.ndarray): Camera to image transformation matrix.
        lidar2cam (np.ndarray): LiDAR to camera transformation matrix.
        visualize_dir (str, optional): If set, saves a per-sample subplot
            of `gt_depths` (one panel per camera) to this directory.
            Useful for debugging the projection. Defaults to None.
        max_depth (float): Upper clip for the depth color scale (m).
            Defaults to 80.
    """

    def __init__(
        self,
        img_shape,
        num_cameras: int,
        depth_bounds: Tuple[float, float],
        visualize_dir: Optional[str] = None,
        max_depth: float = 80.0,
    ):
        self.img_shape = img_shape
        self.num_cameras = num_cameras
        self.visualize_dir = visualize_dir
        self.max_depth = max_depth
        self.depth_bounds = depth_bounds
        self.visualize_dir = Path(visualize_dir) if visualize_dir is not None else None
        if self.visualize_dir is not None:
            self.visualize_dir.mkdir(parents=True, exist_ok=True)
        self._depth_idx = 0

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys:
                - gt_depths (np.ndarray): Ground truth depths in (N, H, W) for (number of cameras, height, width).
        """
        lidar2image = np.asarray(results["lidar2img"])
        img_aug_matrix = np.asarray(results["img_aug_matrix"]) if "img_aug_matrix" in results else np.eye(4)
        cur_coords = results["points"].numpy()[:, :3]

        # inverse lidar aug
        if "lidar_aug_matrix" in results:
            lidar_aug_matrix = np.asarray(results["lidar_aug_matrix"])
            lidar_aug_matrix_inverse = np.linalg.inv(lidar_aug_matrix)
            cur_coords -= lidar_aug_matrix[:3, 3]
            cur_coords = lidar_aug_matrix_inverse[:3, :3] @ cur_coords.transpose(1, 0)
        else:
            cur_coords = cur_coords.transpose(1, 0)

        # lidar2image
        cur_coords = lidar2image[:, :3, :3] @ cur_coords
        cur_coords += lidar2image[:, :3, 3].reshape(-1, 3, 1)

        # get 2d coords
        dist = cur_coords[:, 2, :]
        valid_dist_mask = (dist >= self.depth_bounds[0]) & (dist < self.depth_bounds[1])

        cur_coords[:, 2, :] = np.clip(cur_coords[:, 2, :], 1e-5, 1e5)
        cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

        # imgaug
        cur_coords = img_aug_matrix[:, :3, :3] @ cur_coords
        cur_coords += img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
        cur_coords = cur_coords[:, :2, :].transpose(0, 2, 1)

        # normalize coords for grid sample
        cur_coords = cur_coords[..., [1, 0]]
        on_img = (
            (cur_coords[..., 0] < self.img_shape[0])
            & (cur_coords[..., 0] >= 0)
            & (cur_coords[..., 1] < self.img_shape[1])
            & (cur_coords[..., 1] >= 0)
            & valid_dist_mask
        )

        # Avoid loops since it's slow
        indices = np.nonzero(on_img)
        camera_indices = indices[0]
        point_indices = indices[1]
        masked_coords = cur_coords[camera_indices, point_indices].astype(np.int64)
        masked_dist = dist[camera_indices, point_indices]

        # Possibly to have duplicates and the last one will be used, however, the chance is small
        flatten_indices = (
            camera_indices * self.img_shape[0] * self.img_shape[1]
            + masked_coords[:, 0] * self.img_shape[1]
            + masked_coords[:, 1]
        )
        depth_flat = np.zeros(self.num_cameras * self.img_shape[0] * self.img_shape[1], dtype=np.float32)
        depth_flat[flatten_indices] = masked_dist
        depth = depth_flat.reshape(self.num_cameras, self.img_shape[0], self.img_shape[1])
        results["gt_depths"] = depth

        if self.visualize_dir is not None:
            self._save_depth_subplot(depth, results)
        return results

    def _save_depth_subplot(self, depth: np.ndarray, results: dict) -> None:
        """Save `gt_depths` as a subplot with one panel per camera.

        The figure contains three row blocks per camera:
        - image underlay (if available) + projected LiDAR depth points
        - image pixels only
        - depth-only heatmap (no image pixel values)

        Args:
            depth (np.ndarray): (num_cameras, H, W) ground-truth depth map.
            results (dict): The pipeline result dict; used for the underlay
                image and to derive a unique filename.
        """
        imgs = results.get("img", None)

        # Layout:
        # - Top block: image underlay + projected depth points.
        # - Middle block: image pixels only.
        # - Bottom block: depth-only heatmap (no image pixel values).
        if self.num_cameras <= 6:
            base_rows, cols = 1, self.num_cameras
        else:
            cols = int(np.ceil(np.sqrt(self.num_cameras)))
            base_rows = int(np.ceil(self.num_cameras / cols))
        rows = base_rows * 3

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

        for c in range(self.num_cameras):
            d = depth[c]
            ys, xs = np.nonzero(d)
            vals = d[ys, xs]

            # Row block 1: image + depth scatter.
            ax_overlay = axes[c // cols, c % cols]
            if imgs is not None and c < len(imgs):
                ax_overlay.imshow(imgs[c].astype(np.uint8))
                if vals.size > 0:
                    ax_overlay.scatter(
                        xs,
                        ys,
                        c=vals,
                        cmap="turbo",
                        vmin=0,
                        vmax=self.max_depth,
                        s=1,
                    )
            else:
                ax_overlay.imshow(
                    d,
                    cmap="turbo",
                    vmin=0,
                    vmax=self.max_depth,
                    interpolation="nearest",
                )
            ax_overlay.set_title(f"cam {c} overlay  ({vals.size} pts)")
            ax_overlay.set_xticks([])
            ax_overlay.set_yticks([])

            # Row block 2: image-only visualization.
            ax_img = axes[base_rows + (c // cols), c % cols]
            if imgs is not None and c < len(imgs):
                ax_img.imshow(imgs[c].astype(np.uint8))
            else:
                ax_img.imshow(
                    d,
                    cmap="gray",
                    vmin=0,
                    vmax=self.max_depth,
                    interpolation="nearest",
                )
            ax_img.set_title(f"cam {c} image-only")
            ax_img.set_xticks([])
            ax_img.set_yticks([])

            # Row block 3: depth-only visualization.
            ax_depth = axes[(base_rows * 2) + (c // cols), c % cols]
            ax_depth.imshow(
                d,
                cmap="turbo",
                vmin=0,
                vmax=self.max_depth,
                interpolation="nearest",
            )
            ax_depth.set_title(f"cam {c} depth-only")
            ax_depth.set_xticks([])
            ax_depth.set_yticks([])

        # Hide any unused subplots when n doesn't fill the grid.
        for c in range(self.num_cameras, base_rows * cols):
            axes[c // cols, c % cols].axis("off")
            axes[base_rows + (c // cols), c % cols].axis("off")
            axes[(base_rows * 2) + (c // cols), c % cols].axis("off")

        # Shared depth colorbar with numeric values.
        depth_mappable = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin=0, vmax=self.max_depth))
        depth_mappable.set_array([])
        cbar = fig.colorbar(depth_mappable, ax=axes, location="right", fraction=0.02, pad=0.02)
        cbar.set_label("Depth (m)")

        fig.suptitle(f"gt_depths — {self._depth_idx}")
        fig.tight_layout(rect=[0, 0, 0.96, 0.97])

        self._depth_idx += 1
        out_path = self.visualize_dir / f"{self._depth_idx:06d}_gt_depths.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved gt_depths visualization to {out_path}")
