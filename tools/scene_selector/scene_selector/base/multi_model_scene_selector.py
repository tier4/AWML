from typing import Dict, List, Union

import numpy as np

from .scene_selector import SceneSelector


class ImagePointcloudSceneSelector(SceneSelector):

    def __init__() -> None:
        pass

    def is_target_scene(
        self,
        sensor_info: Union[List[Dict], Dict],
        *args,
        **kwargs,
    ) -> List[bool]:
        """
        Determines whether the given scene is a target scene.

        Args:
            sensor_info (Union[List[Dict], Dict]): Sensor information of the scene.

                If `sensor_info` is a `Dict`, it should have the following structure:

                ```python
                {
                    'images': {
                        'CAMERA_NAME': {
                            'img_path': str,             # Path to the image file
                            'cam2img': List[List[float]], # 3x3 camera intrinsic matrix
                            'sample_data_token': str,    # Unique identifier for the sample data
                            'timestamp': float,          # Timestamp of the image capture
                            'cam2ego': List[List[float]], # 4x4 transformation matrix from camera to ego vehicle
                            'lidar2cam': List[List[float]] # 4x4 transformation matrix from LiDAR to camera
                        },
                        ...
                    },
                    'points': str  # Path to the point cloud data file
                }
                ```

                - **'images'**: A dictionary where each key is a camera name (e.g., 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_BACK_RIGHT') and each value is another dictionary containing camera data.
                - **'points'**: A string representing the path to the point cloud data file.

                Each camera dictionary contains:

                - **'img_path'** (`str`): Path to the image file.
                - **'cam2img'** (`List[List[float]]`): 3x3 intrinsic camera matrix.
                - **'sample_data_token'** (`str`): Unique identifier for the sample data.
                - **'timestamp'** (`float`): Timestamp of the image capture.
                - **'cam2ego'** (`List[List[float]]`): 4x4 extrinsic matrix from camera to ego vehicle coordinate system.
                - **'lidar2cam'** (`List[List[float]]`): 4x4 transformation matrix from LiDAR to camera coordinate system.
        Returns:
            bool: List[`True`] if the scene matches the target criteria, `False` otherwise.
        """
        pass
