from typing import List

import numpy as np
import numpy.typing as npt
import open3d
import open3d.visualization

PALETTE = [
    ([255, 120, 50], "orange"),  # barrier
    ([255, 192, 203], "pink"),  # bicycle
    ([255, 255, 0], "yellow"),  # bus
    ([0, 150, 245], "blue"),  # car
    ([0, 255, 255], "cyan"),  # construction_vehicle
    ([255, 127, 0], "dark orange"),  # motorcycle
    ([255, 0, 0], "red"),  # pedestrian
    ([255, 240, 150], "light yellow"),  # traffic_cone
    ([135, 60, 0], "brown"),  # trailer
    ([160, 32, 240], "purple"),  # truck
    ([255, 0, 255], "dark pink"),  # driveable_surface
    ([139, 137, 137], "dark red"),  # other_flat
    ([75, 0, 75], "dark purple"),  # sidewalk
    ([150, 240, 80], "light green"),  # terrain
    ([230, 230, 250], "white"),  # manmade
    ([0, 175, 0], "green"),  # vegetation
    ([0, 0, 0], "black"),  # unknown
]


class Visualizer:

    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names + ["unknown"]

    def visualize(
        self,
        batch_inputs_dict: dict,
        predictions: npt.ArrayLike,
        num_points: int = -1,
    ) -> None:
        predictions = predictions[:num_points]

        unique_values, counts = np.unique(predictions, return_counts=True)
        print(f"Predictions of total {predictions.shape[0]} points:")
        for key, value in enumerate(unique_values):
            print(f"{self.class_names[value]} - {counts[key]} points ({PALETTE[value][1]})")

        points = batch_inputs_dict["points"][:num_points]
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(points[:, :3])
        colors = np.array([PALETTE[i][0] for i in predictions])
        point_cloud.colors = open3d.utility.Vector3dVector(colors / 255.0)

        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        vis.add_geometry(point_cloud)
        vis.run()
        vis.destroy_window()
