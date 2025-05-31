import argparse
import logging
import os
import os.path as osp
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List
import shutil

import mmengine
import numpy as np
import yaml
from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from mmengine.config import Config
from mmengine.logging import print_log
from t4_devkit import Tier4
from t4_devkit.common.timestamp import us2sec
from t4_devkit.schema import Sample

from tools.detection3d.t4dataset_converters.t4converter import (
    extract_tier4_data,
    get_annotations,
    get_ego2global,
    get_lidar_points_info,
    get_lidar_sweeps_info,
    obtain_sensor2top,
    parse_camera_path,
)
from tools.detection3d.t4dataset_converters.update_infos_to_v2 import get_empty_standard_data_info



def parse_args():
    parser = argparse.ArgumentParser(description="Create data info for T4dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config for T4dataset",
    )
    args = parser.parse_args()
    return args


def main():
    destination = "data/t4dataset/db_j6gen2_v2/"
    os.makedirs(destination, exist_ok=True)
    args = parse_args()
    
    with open(args.config, "r") as f:
      dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)
    
    source  = "data/t4dataset/db_j6gen2_v1/"
    dataset_list_dict.pop("version")
    dataset_list_dict.pop("dataset_version")
    for split, datasets in dataset_list_dict.items():
      for dataset in datasets:
        shutil.move(source + dataset, destination)

if __name__ == "__main__":
    main()
