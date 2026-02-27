"""Point-cloud visualizer for FRNet deployment.

Shows a 3D point cloud coloured by predicted semantic class using Open3D.
"""

from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt
import open3d
import open3d.visualization
from mmengine.logging import MMLogger


class Visualizer:
    """Semantic segmentation point-cloud visualizer."""

    def __init__(self, class_names: List[str], palette: List[List[int]]) -> None:
        self._class_names = class_names
        self._palette = palette
        self.logger = MMLogger.get_current_instance()

    def visualize(
        self,
        batch_inputs_dict: dict,
        predictions: npt.NDArray[np.intp],
        num_points: int = -1,
    ) -> None:
        """Log per-class summary and open interactive Open3D viewer."""
        if num_points > 0:
            predictions = predictions[:num_points]

        unique_values, counts = np.unique(predictions, return_counts=True)
        self.logger.info(f"Predictions of total {predictions.shape[0]} points:")
        for value, count in zip(unique_values, counts):
            name = self._class_names[value] if value < len(self._class_names) else f"class_{value}"
            self.logger.info(f"  {name} - {count} points")

        points = batch_inputs_dict["points"]
        if num_points > 0:
            points = points[:num_points]

        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(points[:, :3])
        colors = np.array([self._palette[i] if i < len(self._palette) else [128, 128, 128] for i in predictions])
        point_cloud.colors = open3d.utility.Vector3dVector(colors / 255.0)

        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        vis.add_geometry(point_cloud)
        vis.run()
        vis.destroy_window()
