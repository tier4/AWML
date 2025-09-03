import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from mmdet.datasets import BaseDetDataset, CocoDataset
from mmdet.registry import DATASETS


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
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not valid JSON.")

    return {}


@DATASETS.register_module()
class T4Dataset(BaseDetDataset):

    def __init__(
        self,
        test_mode=False,
        *args,
        **kwargs,
    ) -> None:

        self.test_mode = test_mode
        super().__init__(test_mode=test_mode, *args, **kwargs)

    def load_data_list(self) -> List[dict]:
        # Call the superclass method to load the data list
        data_list = super().load_data_list()
        # If not in evaluation mode, return the data list as is
        if not self.test_mode:
            return data_list

        img_id = 0
        for data_info in data_list:
            data_info["img_id"] = img_id
            img_id += 1

        return data_list
