import argparse
import logging
import os
import os.path as osp
import re
import warnings
from typing import Any, Dict, List

import mmengine
import yaml
from mmengine.config import Config
from mmengine.logging import print_log
from nuscenes import NuScenes

from tools.detection3d.t4dataset_converters.t4converter import (
    extract_nuscenes_data, get_annotations, get_ego2global,
    get_lidar_points_info, get_lidar_sweeps_info, obtain_sensor2top)
from tools.detection3d.t4dataset_converters.update_infos_to_v2 import \
    get_empty_standard_data_info


def get_lidar_token(sample_rec: Dict[str, Dict[str, str]]) -> str:
    data_dict = sample_rec["data"]
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
        dataset_version (str): The dataset version like 'database_v1_1'
        scene_id: The scene id token.
    Returns:
        str: The updated path containing the version directory if it exists,
            otherwise the original `scene_root_dir_path`.
    """
    # an integer larger than or equal to 0
    version_pattern = re.compile(r"^\d+$")

    # "./data/t4dataset/database_v1_1/e6d0237c-274c-4872-acc9-dc7ea2b77943"
    scene_root_dir_path = osp.join(root_path, dataset_version, scene_id)

    version_dirs = [
        d for d in os.listdir(scene_root_dir_path) if version_pattern.match(d)
    ]

    if version_dirs:
        version_id = sorted(version_dirs, key=int)[-1]
        # "./data/t4dataset/database_v1_1/e6d0237c-274c-4872-acc9-dc7ea2b77943/0"
        return os.path.join(scene_root_dir_path, version_id)
    else:
        warnings.simplefilter("always")
        warnings.warn(
            f"The directory structure of T4 Dataset is deprecated. In the newer version, the directory structure should look something like `$T4DATASET_ID/$VERSION_ID/`. Please update your Web.Auto CLI to the latest version.",
            DeprecationWarning)
        return scene_root_dir_path


def get_info(cfg: Any, nusc: Any, sample: Any, i: int, max_sweeps: int):
    lidar_token = get_lidar_token(sample)
    if lidar_token is None:
        print_log(
            f"sample {sample['token']} doesn't have lidar",
            level=logging.WARNING,
        )
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
    ) = extract_nuscenes_data(nusc, sample, lidar_token)

    info = get_empty_standard_data_info(cfg.camera_types)

    basic_info = dict(
        sample_idx=i,
        token=sample["token"],
        timestamp=sample["timestamp"] / 1e6,
        scene_token=sample["scene_token"],
        location=log_record["location"],
        scene_name=scene_record["name"],
    )

    for new_info in [
            basic_info,
            get_ego2global(pose_record),
            get_lidar_points_info(lidar_path, cs_record),
            get_lidar_sweeps_info(nusc, cs_record, pose_record, sd_record,
                                  max_sweeps),
            get_annotations(
                nusc,
                sample["anns"],
                boxes,
                e2g_r_mat,
                l2e_r_mat,
                cfg.name_mapping,
                cfg.class_names,
            ),
    ]:
        info.update(new_info)

    camera_types = cfg.camera_types
    if (len(camera_types) > 0):
        for cam in camera_types:
            #print(lidar_path, sample['data'])
            if cam in sample['data']:
                cam_token = sample['data'][cam]
                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                             e2g_t, e2g_r_mat, cam)
                cam_info.update(cam_intrinsic=cam_intrinsic)
                info.update({cam: cam_info})

    return info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create data info for T4dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config for T4dataset",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="product version",
    )
    parser.add_argument(
        "--max_sweeps",
        type=int,
        required=True,
        help="specify sweeps of lidar per example",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="output directory of info file",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    t4_infos = {
        "train": [],
        "val": [],
        "test": [],
    }
    metainfo = dict(classes=cfg.class_names, version=args.version)

    for dataset_version in cfg.dataset_version_list:
        dataset_list = osp.join(cfg.dataset_version_config_root,
                                dataset_version + ".yaml")
        with open(dataset_list, "r") as f:
            dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)

        for split in ["train", "val", "test"]:
            print_log(
                f"Creating data info for split: {split}", logger="current")
            for scene_id in dataset_list_dict.get(split, []):
                scene_root_dir_path = get_scene_root_dir_path(
                    args.root_path,
                    dataset_version,
                    scene_id,
                )

                if not osp.isdir(scene_root_dir_path):
                    raise ValueError(f"{scene_root_dir_path} does not exist.")
                nusc = NuScenes(
                    version="annotation",
                    dataroot=scene_root_dir_path,
                    verbose=False)
                for i, sample in enumerate(nusc.sample):
                    info = get_info(cfg, nusc, sample, i, args.max_sweeps)
                    t4_infos[split].append(info)
    assert sum(len(split)
               for split in t4_infos.values()) > 0, "dataset isn't available"
    print(f"train sample: {len(t4_infos['train'])}, "
          f"val sample: {len(t4_infos['val'])}, "
          f"test sample: {len(t4_infos['test'])}")

    def save(_infos, _split):
        _info_path = osp.join(args.out_dir,
                              f"t4dataset_{args.version}_infos_{_split}.pkl")
        mmengine.dump(dict(data_list=_infos, metainfo=metainfo), _info_path)

    save(t4_infos["train"], "train")
    save(t4_infos["val"], "val")
    save(t4_infos["train"] + t4_infos["val"], "trainval")
    save(t4_infos["test"], "test")
    save(t4_infos["train"] + t4_infos["val"] + t4_infos["test"], "all")


if __name__ == "__main__":
    main()
