# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
from mmcv.transforms.base import BaseTransform

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class IntensityMinMaxNorm(BaseTransform):
    """
    """

    def __init__(self,
                 intensity_dim: int = 3) -> None:
        super().__init__()
        self._intensity_dim = intensity_dim
        self._max_value = 255
        self._min_value	= 0
        self._range = self._max_value - self._min_value

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        points = results['points']
        point_intensity = points.tensor[:, self._intensity_dim]
        point_intensity = np.clip(point_intensity, self._min_value, self._max_value)
        point_intensity = (point_intensity - self._min_value) / self._range
        points.tensor[:, self._intensity_dim] = point_intensity
        results['points'] = points
        return results
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'intensity_dim={self._intensity_dim}, '
        repr_str += f'min_value={self._min_value}, '
        repr_str += f'value_value={self._max_value})'
        return repr_str



@TRANSFORMS.register_module()
class IntensityStandardization(BaseTransform):
    """
    """

    def __init__(self,
                 mean: float, 
                 std: float,
                 intensity_dim: int = 3, 
                 ) -> None:
        super().__init__()
        self._intensity_dim = intensity_dim
        self._mean = mean
        self._std = std

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        points = results['points']
        point_intensity = points.tensor[:, self._intensity_dim]
        point_intensity = (point_intensity - self._mean) / self._std
        points.tensor[:, self._intensity_dim] = point_intensity
        results['points'] = points
        return results
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'intensity_dim={self._intensity_dim}, '
        repr_str += f'mean={self._mean}, '
        repr_str += f'std={self._std})'
        return repr_str