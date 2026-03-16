import torch
import numpy as np
from mmdet3d.structures import LiDARPoints
from autoware_ml.detection3d.datasets.transforms.point_offset import PointOffset

def test_point_offset():
    # 5 points, each with [x, y, z, intensity, sensor_id]
    points_data = np.array([
        [1.0, 1.0, 1.0, 0.5, 0],
        [2.0, 2.0, 2.0, 0.6, 1],
        [3.0, 3.0, 3.0, 0.7, 0],
        [4.0, 4.0, 4.0, 0.8, 1],
        [5.0, 5.0, 5.0, 0.9, 2]
    ], dtype=np.float32)
    
    points = LiDARPoints(torch.from_numpy(points_data), points_dim=5)
    input_dict = {'points': points}
    
    # Offset only sensor_id=1 by [0.1, -0.2, 0.5]
    offset = [0.1, -0.2, 0.5]
    transform = PointOffset(offset=offset, sensor_id=1, sensor_dim=4)
    
    output_dict = transform.transform(input_dict)
    transformed_points = output_dict['points'].tensor.numpy()
    
    # Expected results
    expected_points = points_data.copy()
    expected_points[1, :3] += offset
    expected_points[3, :3] += offset
    
    assert np.allclose(transformed_points, expected_points), f"Expected {expected_points}, but got {transformed_points}"
    print("Test sensor_id filter: PASSED")

    # Test offsetting all points
    points = LiDARPoints(torch.from_numpy(points_data), points_dim=5)
    input_dict = {'points': points}
    transform_all = PointOffset(offset=offset, sensor_id=None)
    output_dict_all = transform_all.transform(input_dict)
    transformed_all = output_dict_all['points'].tensor.numpy()
    
    expected_all = points_data.copy()
    expected_all[:, :3] += offset
    
    assert np.allclose(transformed_all, expected_all), "Test all points failed"
    print("Test all points: PASSED")

if __name__ == "__main__":
    test_point_offset()
