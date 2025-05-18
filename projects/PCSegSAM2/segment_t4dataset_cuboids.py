import argparse
import concurrent.futures
import logging
import os
import os.path as osp
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from mmengine.config import Config
from t4_devkit import Tier4
from t4_devkit.schema import Sample
from tqdm import tqdm

from tools.detection3d.t4dataset_converters.t4converter import (
    extract_tier4_data,
)


def get_lidar_token(sample_rec: Sample) -> str:
    data_dict = sample_rec.data
    if "LIDAR_TOP" in data_dict:
        return data_dict["LIDAR_TOP"]
    elif "LIDAR_CONCAT" in data_dict:
        return data_dict["LIDAR_CONCAT"]
    else:
        return None


def get_scene_root_dir_path(
    root_path: str,
    dataset_version: str,
    scene_id: str,
) -> str:
    """
    This function checks if the provided `scene_root_dir_path` follows the new directory structure
    of the T4 Dataset, which should look like `$T4DATASET_VERSION/$T4DATASET_ID/$VERSION_ID/`.
    If the `scene_root_dir_path` does contain a version directory, it searches for the latest version directory
    under the `scene_root_dir_path` and returns the updated path.
    If no version directory is found, it prints a deprecation warning and returns the original `scene_root_dir_path`.

    Args:
        root_path (str): The root path of the T4 Dataset.
        dataset_version (str): The dataset version like 'db_jpntaxi_v2'
        scene_id: The scene id token.
    Returns:
        str: The updated path containing the version directory if it exists,
            otherwise the original `scene_root_dir_path`.
    """
    # an integer larger than or equal to 0
    version_pattern = re.compile(r"^\d+$")

    scene_root_dir_path = osp.join(root_path, dataset_version, scene_id)

    version_dirs = [d for d in os.listdir(scene_root_dir_path) if version_pattern.match(d)]

    if version_dirs:
        version_id = sorted(version_dirs, key=int)[-1]
        return os.path.join(scene_root_dir_path, version_id)
    else:
        warnings.simplefilter("always")
        warnings.warn(
            f"The directory structure of T4 Dataset is deprecated. In the newer version, the directory structure should look something like `$T4DATASET_ID/$VERSION_ID/`. Please update your Web.Auto CLI to the latest version.",
            DeprecationWarning,
        )
        return scene_root_dir_path


def segment_pointcloud(
    root_path: str,
    cfg: Any,
    segmentation_cfg: Any,
    t4: Tier4,
    sample: Sample,
    i: int,
):
    lidar_token = get_lidar_token(sample)
    if lidar_token is None:
        logging.warn(f"sample {sample['token']} doesn't have lidar")
        return
    (
        pose_record,
        cs_record,
        sd_record,
        scene_record,
        log_record,
        boxes,
        lidar_path,
        e2g_r_mat,
        l2e_r_mat,
        e2g_t,
        l2e_t,
    ) = extract_tier4_data(t4, sample, lidar_token)

    lidar_l2e_transform = np.eye(4, dtype=np.float32)
    lidar_l2e_transform[0:3, 0:3] = l2e_r_mat
    lidar_l2e_transform[0:3, 3] = l2e_t

    cuboid_segmentation_cfg = segmentation_cfg["cuboid_segmentation"]
    invalid_value = cuboid_segmentation_cfg["invalid_value"]
    reset_classes = cuboid_segmentation_cfg["reset_classes"]
    cuboid_to_segmentation_class_map = cuboid_segmentation_cfg["classes_map"]

    # Load points
    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
    num_points = points.shape[0]
    points_lcs = np.hstack([points[:, 0:3], np.ones((num_points, 1))])
    points_ecs = points_lcs @ lidar_l2e_transform.T
    
    lidar_path = Path(lidar_path)
    basename = lidar_path.name.split(".")[0]
    seg_path = lidar_path.parent / f"{basename}_seg.npy"

    if seg_path.exists():
        seg_pointcloud = np.load(str(seg_path)).reshape([-1])
    else:
        seg_pointcloud = np.full((num_points,), invalid_value, dtype=np.uint8)

    if len(boxes) > 0:
        for idx in reset_classes:
            seg_pointcloud[seg_pointcloud == idx] = invalid_value

    # NOTE(knzo25): if the segmentation is slow, this can be easily parallelized
    for box in boxes:
        
        center = box.position
        rotation = box.rotation.rotation_matrix

        transform = np.eye(4, dtype=np.float32)
        transform[0:3, 0:3] = rotation
        transform[0:3, 3] = center
        transform = np.linalg.inv(transform)

        points_box = points_ecs @ transform.T
        shape = box.shape.size

        mask = np.logical_and.reduce(
            (
                np.abs(points_box[:, 0]) <= 0.5*shape[1],
                np.abs(points_box[:, 1]) <= 0.5*shape[0],
                np.abs(points_box[:, 2]) <= 0.5*shape[2],
            )
        )

        segmentation_idx = cuboid_to_segmentation_class_map[box.semantic_label.name]
        seg_pointcloud[mask] = segmentation_idx

    try:
        with open(str(seg_path), "wb") as f:
            np.save(f, seg_pointcloud.astype(np.uint8))
    except Exception as e:
        logging.error(f"Failed to save segmentation file {str(seg_path)}: {e}")

    return

def segment_scene(args, cfg, segmentation_cfg, dataset_version, scene_id):

    logging.info(f"Segmenting pointclouds from scene: {scene_id}")
    scene_root_dir_path = get_scene_root_dir_path(
        args.root_path,
        dataset_version,
        scene_id,
    )

    if not osp.isdir(scene_root_dir_path):
        raise ValueError(f"{scene_root_dir_path} does not exist.")

    t4 = Tier4(version="annotation", data_root=scene_root_dir_path, verbose=False)

    for i, sample in enumerate(tqdm(t4.sample)):
        segment_pointcloud(args.root_path, cfg, segmentation_cfg, t4, sample, i)


def parse_args():
    parser = argparse.ArgumentParser(description="Create data info for T4dataset")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config for T4dataset",
    )

    parser.add_argument(
        "--segmentation_config",
        type=str,
        required=True,
        help="config for segmentation",
    )

    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="specify the root path of dataset",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    logging.basicConfig(level=logging.INFO)

    # TODO(knzo25): hack since I only want to test part of the db
    cfg.dataset_version_list = ["db_jpntaxi_v2"]

    with open(args.segmentation_config, "r") as f:
        segmentation_cfg = yaml.safe_load(f)

    num_workers = segmentation_cfg["projective_segmentation"]["num_workers"]

    for dataset_version in cfg.dataset_version_list:
        dataset_list = osp.join(cfg.dataset_version_config_root, dataset_version + ".yaml")
        with open(dataset_list, "r") as f:
            dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)

        for split in ["train", "val", "test"]:
            logging.info(f"Segmenting split: {split}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                executor.map(
                    lambda scene_id: segment_scene(args, cfg, segmentation_cfg, dataset_version, scene_id),
                    dataset_list_dict.get(split, []),
                )


if __name__ == "__main__":
    main()
