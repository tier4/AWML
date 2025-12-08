"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import open3d as o3d


def create_colors_from_predictions(predictions, num_points, class_colors, logger=None):
    """Create RGB colors from predictions using class color mapping.

    Args:
        predictions: Array of prediction class indices
        num_points: Number of points to color
        class_colors: Class colors dictionary
        logger: Logger for messages
    Returns:
        colors: RGB colors as uint8 array (num_points, 3)
    """

    colors = np.zeros((num_points, 3), dtype=np.uint8)
    for i, pred_class in enumerate(predictions):
        if pred_class < len(class_colors):
            colors[i] = class_colors[pred_class]
        else:
            colors[i] = (255, 255, 255)  # white for unknown classes
            if logger:
                logger.warning(f"Class {pred_class} not found in class colors")

    return colors


def visualize_point_cloud(coords, colors, title):
    """Create and display/save point cloud using Open3D.

    Args:
        coords: Point coordinates (N, 3)
        colors: RGB colors (N, 3) as uint8
        title: Title of the point cloud
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords[:, :3])  # Use first 3 coordinates (x, y, z)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)  # Normalize to [0, 1]

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=title,
        width=1200,
        height=800,
        left=50,
        top=50,
        point_show_normal=False,
    )


def get_segmentation_colors(labels, result_path, class_colors, logger=None):
    """Visualize segmentation results from saved data.

    Args:
        labels: Labels array
        result_path: Path to saved .npz file with coordinates
        class_colors: Class colors dictionary
        logger: Logger for messages

    Returns:
        coords: Coordinates array
        colors: RGB colors array
    """
    if logger:
        logger.info(f"Visualizing segmentation results")
        logger.info(f"Labels shape: {labels.shape}")

    # Load coordinates from saved result file
    if logger:
        print(f"Loading coordinates from {result_path}")
    result_data = np.load(result_path)
    feat = result_data["feat"]
    coords = feat[:, :3]  # x, y, z coordinates
    if logger:
        logger.info(f"Loaded coordinates from {result_path}, shape: {coords.shape}")

    # Create color mapping from predictions
    colors = create_colors_from_predictions(labels, coords.shape[0], class_colors, logger)

    return coords, colors
