import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS


def project_to_image(points, lidar2cam, cam2img):
    """Transform points from LiDAR to image coordinates."""
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = np.dot(lidar2cam, points_hom.T).T

    # Filter points behind the camera
    valid_mask = points_cam[:, 2] > 0

    points_img = np.dot(cam2img, points_cam[:, :3].T).T
    points_img /= points_img[:, 2:3]
    return points_img[:, :2], valid_mask


def compute_bbox_and_centers(lidar2cam, cam2img, bboxes, labels, img_shape):
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

    C, H, W = img_shape
    # Initialize lists to store valid results
    valid_bboxes_2d = []
    valid_projected_centers = []
    valid_image_depth = []
    valid_labels_list = []

    # Loop through each bounding box
    for bbox_std, bbox, label in zip(bboxes, bboxes.corners, labels):
        # Project corners to image
        center_3d_lidar = bbox_std[:3].numpy()
        corners_img, valid_mask = project_to_image(
            np.concatenate([bbox, bbox.mean(0).reshape(1, 3)]), lidar2cam, cam2img
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


def check_bbox_visibility_in_image(lidar2cam, cam2img, bboxes, labels, img_shape, visibility=0.1):
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
class StreamPETRLoadAnnotations2D(BaseTransform):

    def transform(self, results):

        all_bboxes_2d, all_centers_2d, all_depths, all_labels = [], [], [], []

        for i, k in enumerate(results["images"]):
            bboxes_2d, projected_centers, depths, valid_labels = compute_bbox_and_centers(
                results["extrinsics"][i],
                results["intrinsics"][i],
                results["gt_bboxes_3d"],
                results["gt_labels_3d"],
                results["img"][i].shape,
            )
            all_bboxes_2d.append(bboxes_2d)
            all_centers_2d.append(projected_centers)
            all_depths.append(depths)
            all_labels.append(valid_labels)
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
        for i, k in enumerate(results["images"]):
            is_visible = check_bbox_visibility_in_image(
                results["extrinsics"][i],
                results["intrinsics"][i],
                results["gt_bboxes_3d"],
                results["gt_labels_3d"],
                results["img"][i].shape,
            )
            visibility_mask.append(is_visible)
        visibility_mask = np.stack(visibility_mask).mean(0)

        return results
