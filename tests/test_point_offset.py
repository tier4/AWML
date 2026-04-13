import numpy as np
import torch
from mmdet3d.structures import LiDARPoints
from scipy.spatial.transform import Rotation as R

from autoware_ml.detection3d.datasets.transforms.point_offset import PointOffset


def test_point_offset_with_rotation():
    # 5 points, each with [x, y, z, intensity, sensor_id]
    points_data = np.array(
        [
            [1.0, 0.0, 0.0, 0.5, 0],
            [0.0, 1.0, 0.0, 0.6, 1],
            [0.0, 0.1, 1.0, 0.7, 0],
            [1.0, 1.0, 1.0, 0.8, 1],
            [0.0, 0.0, 0.0, 0.9, 2],
        ],
        dtype=np.float32,
    )

    # 1. Test pure rotation (Yaw 90 degrees) for sensor_id=1
    # Rotation matrix for 90 deg yaw: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # Point [0, 1, 0] rotated 90 deg around Z: [-1, 0, 0]
    # Point [1, 1, 1] rotated 90 deg around Z: [-1, 1, 1]

    points = LiDARPoints(torch.from_numpy(points_data.copy()), points_dim=5)
    input_dict = {"points": points}

    transform = PointOffset(offset=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 90.0], sensor_id=1, sensor_dim=4)
    output_dict = transform.transform(input_dict)
    transformed_points = output_dict["points"].tensor.numpy()

    expected_points = points_data.copy()
    expected_points[1, :3] = [-1.0, 0.0, 0.0]
    expected_points[3, :3] = [-1.0, 1.0, 1.0]

    assert np.allclose(
        transformed_points, expected_points, atol=1e-6
    ), f"Rotation test failed. Got:\n{transformed_points}"
    print("Test sensor_id rotation: PASSED")

    # 2. Test combined rotation and translation for all points
    # Yaw 90 deg + Offset [1, 1, 1]
    points = LiDARPoints(torch.from_numpy(points_data.copy()), points_dim=5)
    input_dict = {"points": points}

    offset = [1.0, 2.0, 3.0]
    rotation = [0.0, 0.0, 90.0]
    transform_all = PointOffset(offset=offset, rotation=rotation, sensor_id=None)
    output_dict_all = transform_all.transform(input_dict)
    transformed_all = output_dict_all["points"].tensor.numpy()

    rot_mat = R.from_euler("xyz", rotation, degrees=True).as_matrix()
    expected_all = points_data.copy()
    expected_all[:, :3] = points_data[:, :3] @ rot_mat.T + offset

    assert np.allclose(transformed_all, expected_all, atol=1e-6), "Combined transform test failed"
    print("Test all points combined: PASSED")


if __name__ == "__main__":
    test_point_offset_with_rotation()
