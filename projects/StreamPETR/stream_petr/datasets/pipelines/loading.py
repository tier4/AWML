from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import LoadAnnotations3D
import torch

import numpy as np


def project_to_image(points, lidar2cam, cam2img):
    """Transform points from LiDAR to image coordinates."""
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = np.dot(lidar2cam, points_hom.T).T
    
    # Filter points behind the camera
    valid_mask = points_cam[:, 2] > 0
    
    points_img = np.dot(cam2img, points_cam[:, :3].T).T
    points_img /= points_img[:, 2:3]
    return points_img[:, :2], valid_mask

def compute_bbox_and_centers(data_dict, bboxes, labels, img_shape):
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

    lidar2cam = np.array(data_dict['lidar2cam'])
    cam2img = np.array(data_dict['cam2img'])
    H, W, C = img_shape
    
    # Initialize lists to store valid results
    valid_bboxes_2d = []
    valid_projected_centers = []
    valid_image_depth = []
    valid_labels_list = []

    # Loop through each bounding box
    for bbox_std, bbox, label in zip(bboxes, bboxes.corners, labels):
        # Project corners to image
        center_3d_lidar = bbox_std[:3].numpy()
        corners_img, valid_mask = project_to_image(np.concatenate([bbox,bbox.mean(0).reshape(1,3)]), lidar2cam, cam2img)
        projected_center = corners_img[-1]
        
        corners_img = corners_img[:-1][valid_mask[:-1]]
        
        
        if len(corners_img) == 0:  # Skip if no corners are visible
            continue
            
        # Compute 2D bbox
        x_min, y_min = np.min(corners_img, axis=0)
        x_max, y_max = np.max(corners_img, axis=0)
        
        # Check if bbox is within image boundaries
        if (x_min >= W or x_max <= 0 or y_min >= H or y_max <= 0):
            continue
            
        # Clip to image boundaries
        x_min = np.clip(x_min, 0, W)
        x_max = np.clip(x_max, 0, W)
        y_min = np.clip(y_min, 0, H)
        y_max = np.clip(y_max, 0, H)
        
        # Store valid results
        valid_bboxes_2d.append([x_min, y_min, x_max, y_max])
        valid_projected_centers.append(projected_center)
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

    return bboxes_2d, projected_centers, object_depth,  valid_labels
    
      
@TRANSFORMS.register_module()
class StreamPETRLoadAnnotations3D(LoadAnnotations3D):
  """
    Overrides some of the methods to make the 
  """
  def _load_2d_infos(self,results):
      all_bboxes_2d, all_centers_2d, all_depths, all_labels, extrinsics, intrinsics = [],[],[],[],[],[]
      for i,k in enumerate(results["images"]):
        bboxes_2d, projected_centers, depths, valid_labels = compute_bbox_and_centers(
            results["images"][k], 
            results["ann_info"]["gt_bboxes_3d"], 
            results["ann_info"]["gt_labels_3d"],
            results["img"][i].shape
        )
        all_bboxes_2d.append(bboxes_2d)
        all_centers_2d.append(projected_centers)
        all_depths.append(depths)
        all_labels.append(valid_labels)
        extrinsics.append(np.array(results["images"][k]["lidar2cam"])[:3,:])
        intrinsics.append(np.array(results["images"][k]["cam2img"]))
        
      results['depths'] = all_depths
      results['centers_2d'] = all_centers_2d
      results['gt_bboxes'] = all_bboxes_2d
      results['gt_bboxes_labels'] = all_labels
      results['intrinsics'] = intrinsics
      results['extrinsics'] = extrinsics
      
      results['ego_pose'] = np.eye(4)
      results['ego_pose_inv'] = np.eye(4)
      
      return results
  def _load_bboxes_depth(self, results):
      if 'depths' not in results or 'centers_2d' not in results:
        results = self._load_2d_infos(results)
      return results
      
  def _load_bboxes(self, results):
      if 'gt_bboxes' not in results:
        results = self._load_2d_infos(results)
      return results

  def _load_labels(self, results: dict) -> dict:
      if 'gt_bboxes_labels' not in results:
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
      return results