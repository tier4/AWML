from typing import Dict, Optional, Sequence, Tuple, List

import os
import numpy as np
from mmdet.registry import DATASETS

from mmdet.datasets import BaseDetDataset, CocoDataset
from .pipelines.transforms import RandomCropWithROI
import json


def read_json_file(file_path):
    """
    Read a JSON file and return its contents as a Python dictionary.

    Args:
    file_path (str): The path to the JSON file.

    Returns:
    dict: The contents of the JSON file as a Python dictionary.

    Raises:
    FileNotFoundError: If the specified file is not found.
    json.JSONDecodeError: If the file is not valid JSON.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not valid JSON.")

    return {}


def get_instance_key(path: str, bbox: List[float]):
    x1, y1, x2, y2 = [int(x) for x in bbox]
    return f"{path}_{x1}_{y1}_{x2},{y2}"


@DATASETS.register_module()
class TLRDetectionDataset(BaseDetDataset):

    def __init__(self,
                 crop_size: Tuple[float, float] = (1.0, 20.0),
                 hw_ratio: Tuple[float, float] = (0.5, 1.0),
                 test_mode=False,
                 *args,
                 **kwargs) -> None:

        self.test_mode = test_mode
        self.crop_size = crop_size
        self.hw_ratio = hw_ratio
        super().__init__(test_mode=test_mode, *args, **kwargs)

    def _precompute_crops(
            self,
            data_info: dict,
            instance: Optional[dict] = None) -> Tuple[int, int, int, int]:
        """
        Precompute the crop area for an image in the dataset for test mode.

        Args:
            data_info (dict): Information about the image, including height and width.
            instance (Optional[dict]): Instance information, including bounding box if available.

        Returns:
            Tuple[int, int, int, int]: The computed crop coordinates (y1, x1, y2, x2).
        """
        img_h, img_w = data_info['height'], data_info['width']
        img_path = data_info.get("img_path")

        key = get_instance_key(img_path,
                               instance['bbox'] if instance else [0] * 4)

        if key in self.cached_crops:
            return tuple(self.cached_crops[key])

        if instance:
            bbox = np.array(instance['bbox'])
            crop_coords = RandomCropWithROI.calculate_roi_crop(
                bbox, img_h, img_w, self.crop_size, self.hw_ratio)
        else:
            crop_coords = RandomCropWithROI.calculate_random_crop(img_h, img_w)

        self.cached_crops[key] = crop_coords
        return crop_coords

    def load_data_list(self) -> List[dict]:
        # Call the superclass method to load the data list
        data_list = super().load_data_list()
        # If not in evaluation mode, return the data list as is
        if not self.test_mode:
            return data_list
        base, fname = os.path.split(self.ann_file)
        fname, _ = fname.split(".")
        precomputed_coords_path = os.path.join(base, f"{fname}_crops.json")
        self.cached_crops = read_json_file(precomputed_coords_path)
        if len(self.cached_crops):
            print(f"TLRDetectionDataset: loaded cropping cache from {precomputed_coords_path}")
        # In evaluation mode, create a new list where each bounding box is treated as a separate instance
        separated_data_list = []
        img_id = 0
        for data_info in data_list:
            instances = data_info.get('instances', [])
            if len(instances) > 0:
                for instance in instances:
                    single_instance_info = data_info.copy()
                    single_instance_info["img_id"] = img_id
                    single_instance_info[
                        "crop_coords"] = self._precompute_crops(
                            single_instance_info, instance)
                    img_id += 1
                    separated_data_list.append(single_instance_info)
            else:
                # If there are one or no instances, just add the original data_info
                data_info["img_id"] = img_id
                data_info["crop_coords"] = self._precompute_crops(data_info)
                img_id += 1
                separated_data_list.append(data_info)
        if not os.path.exists(precomputed_coords_path):
            with open(precomputed_coords_path, 'w') as json_file:
                json.dump(self.cached_crops, json_file, indent=4)
            print(f"TLRDetectionDataset: Saved cropping cache to {precomputed_coords_path}")
        return separated_data_list
