import argparse
import logging
import os
import os.path as osp
from typing import Dict, List

import mmengine
import yaml
from mmengine.config import Config
from mmengine.logging import print_log
from nuscenes import NuScenes

from tools.detection3d.t4dataset_converters.t4converter import (
    extract_nuscenes_data, get_annotations, get_ego2global,
    get_lidar_points_info, get_lidar_sweeps_info)
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


if __name__ == "__main__":
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
                scene_dir = osp.join(args.root_path, dataset_version, scene_id)
                if not osp.isdir(scene_dir):
                    raise ValueError(f"{scene_dir} does not exist.")
                nusc = NuScenes(
                    version="annotation", dataroot=scene_dir, verbose=False)

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
                            get_lidar_sweeps_info(nusc, cs_record, pose_record,
                                                  sd_record, args.max_sweeps),
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
