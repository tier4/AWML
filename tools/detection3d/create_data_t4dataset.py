import argparse
import logging
import os
import os.path as osp
from typing import Dict, List, Optional

import mmengine
import yaml
from mmengine.config import Config
from mmengine.logging import print_log
from nuscenes import NuScenes

from tools.detection3d.t4dataset_converters.t4converter import (
    extract_nuscenes_data, get_annotations, get_ego2global,
    get_lidar_points_info, get_lidar_sweeps_info)
from tools.detection3d.t4dataset_converters.t4dataset_converter import \
    get_lidar_token
from tools.detection3d.t4dataset_converters.update_infos_to_v2 import (
    get_empty_lidar_points, get_empty_radar_points,
    get_empty_standard_data_info, get_single_image_sweep)


def get_empty_standard_data_info(camera_types=["CAM0", "CAM1", "CAM2", "CAM3", "CAM4"]):
    data_info = dict(
        # (str): Sample id of the frame.
        sample_idx=None,
        # (str, optional): '000010'
        token=None,
        **get_single_image_sweep(camera_types),
        # (dict, optional): dict contains information
        # of LiDAR point cloud frame.
        lidar_points=get_empty_lidar_points(),
        # (dict, optional) Each dict contains
        # information of Radar point cloud frame.
        radar_points=get_empty_radar_points(),
        # (list[dict], optional): Image sweeps data.
        image_sweeps=[],
        lidar_sweeps=[],
        instances=[],
        # (list[dict], optional): Required by object
        # detection, instance  to be ignored during training.
        instances_ignore=[],
        # (str, optional): Path of semantic labels for each point.
        pts_semantic_mask_path=None,
        # (str, optional): Path of instance labels for each point.
        pts_instance_mask_path=None,
    )
    return data_info


def t4dataset_data_prep(
    root_path: str,
    version: str,
    out_dir: str,
    dataset_version_list: List[str],
    class_names: List[str],
    name_mapping: List[str],
    camera_types: List[str],
    max_sweeps: int = 1,
):
    """Prepare data related to nuScenes dataset.
    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.
    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 1
    """
    t4_infos = {
        "train": [],
        "val": [],
        "test": [],
    }
    metainfo = dict(classes=class_names, version=version)

    for dataset_version in dataset_version_list:
        dataset_version_config_root = "autoware_ml/configs/detection3d/dataset/t4dataset/"
        dataset_list = osp.join(dataset_version_config_root, dataset_version + ".yaml")
        with open(dataset_list, "r") as f:
            dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)

        for split in ["train", "val", "test"]:
            print_log(f"Creating data info for split: {split}", logger="current")
            for scene_id in dataset_list_dict.get(split, []):
                scene_dir = osp.join(root_path, dataset_version, scene_id)
                if not osp.isdir(scene_dir):
                    raise ValueError(f"{scene_dir} does not exist.")

                nusc = NuScenes(version="annotation", dataroot=scene_dir, verbose=False)

                for i, sample in enumerate(nusc.sample):
                    lidar_token = get_lidar_token(sample)
                    if lidar_token is None:
                        print_log(
                            f"sample {sample['token']} doesn't have lidar",
                            level=logging.WARNING,
                        )
                        continue
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
                    ) = extract_nuscenes_data(nusc, sample, lidar_token)

                    info = get_empty_standard_data_info(camera_types)

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
                        get_lidar_sweeps_info(
                            nusc, cs_record, pose_record, sd_record, max_sweeps
                        ),
                        get_annotations(
                            nusc,
                            sample["anns"],
                            boxes,
                            e2g_r_mat,
                            l2e_r_mat,
                            name_mapping,
                            class_names,
                        ),
                    ]:
                        info.update(new_info)
                    t4_infos[split].append(info)
    assert sum(len(split) for split in t4_infos.values()) > 0, "dataset isn't available"
    print(
        f"train sample: {len(t4_infos['train'])}, "
        f"val sample: {len(t4_infos['val'])}, "
        f"test sample: {len(t4_infos['test'])}"
    )

    def save(_infos, _split):
        _info_path = osp.join(out_dir, f"t4dataset_{version}_infos_{_split}.pkl")
        mmengine.dump(dict(data_list=_infos, metainfo=metainfo), _info_path)

    save(t4_infos["train"], "train")
    save(t4_infos["val"], "val")
    save(t4_infos["train"] + t4_infos["val"], "trainval")
    save(t4_infos["test"], "test")
    save(t4_infos["train"] + t4_infos["val"] + t4_infos["test"], "all")


def parse_args():
    parser = argparse.ArgumentParser(description="Create data info for T4dataset")
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

    t4dataset_data_prep(
        root_path=args.root_path,
        version=args.version,
        out_dir=args.out_dir,
        max_sweeps=args.max_sweeps,
        dataset_version_list=cfg.dataset_version_list,
        name_mapping=cfg.name_mapping,
        class_names=cfg.class_names,
        camera_types=cfg.camera_types,
    )


if __name__ == "__main__":
    main()
