"""
nuScenes Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Zheng Zhang
Please cite our work if the code is helpful to you.
"""

import os
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class T4Dataset(DefaultDataset):
    def __init__(self, sweeps=10, ignore_index=-1, **kwargs):
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(self.data_root, "info", f"t4dataset_xx1_infos_train.pkl")
        elif split == "val":
            return os.path.join(self.data_root, "info", f"t4dataset_xx1_infos_val.pkl")
        elif split == "test":
            return os.path.join(self.data_root, "info", f"t4dataset_xx1_infos_test.pkl")
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
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

        lidar_path = Path(lidar_path)
        basename = lidar_path.name.split(".")[0]
        seg_path = lidar_path.parent / f"{basename}_seg.npy"

        segment = np.load(str(seg_path)).reshape([-1])
        segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(np.int64)

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

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            255: ignore_index,
        }
        return learning_map
