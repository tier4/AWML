"""
nuScenes Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Zheng Zhang
Please cite our work if the code is helpful to you.
"""

import os
import pickle
from typing import Dict

import numpy as np
import numpy.typing as npt

from autoware_ml.segmentation3d.datasets.utils import load_and_map_semantic_mask

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class T4Dataset(DefaultDataset):
    def __init__(self, ignore_index=-1, class_mapping={}, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = class_mapping
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_info_paths(self):
        return [os.path.join(self.data_root, path) for path in self.info_paths]

    def get_data_list(self):
        data_list = []
        for info_path in self.get_info_paths():
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info["data_list"])
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, data["lidar_points"]["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]

        lidarseg_path = data["pts_semantic_mask_path"]
        lidarseg_categories = data["pts_semantic_mask_categories"]
        segment = load_and_map_semantic_mask(
            mask_path=str(lidarseg_path),
            raw_categories=lidarseg_categories,
            class_mapping=self.learning_map,
            ignore_index=self.ignore_index,
            seg_dtype=np.uint8,
        )

        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["token"]
