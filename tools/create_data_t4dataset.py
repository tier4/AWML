import argparse
import logging
import os
import os.path as osp
from typing import Dict, List, Optional

import mmengine
from mmengine.logging import print_log
from nuscenes import NuScenes
import yaml

from autoware_ml.detection.datasets import camera_types, classes
from autoware_ml.detection.datasets.name_mappings import get_mapping
from tools.t4dataset_converters.t4converter import (
    extract_nuscenes_data,
    get_annotations,
    get_ego2global,
    get_lidar_points_info,
    get_lidar_sweeps_info,
)
from tools.t4dataset_converters.t4dataset_converter import get_lidar_token
from tools.t4dataset_converters.update_infos_to_v2 import (
    get_empty_lidar_points,
    get_empty_radar_points,
    get_empty_standard_data_info,
    get_single_image_sweep,
)


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
    info_prefix: str,
    version: str,
    out_dir: str,
    max_sweeps: int = 1,
    dataset_config: Optional[str] = None,
    overwrite_root_path: Optional[str] = None,
    use_2d_annotation=False,
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
    assert any(
        [
            version
            in (
                "t4xx1",
                "t4x2",
                "t4x2_awsim",
                "t4xx1_uc2",
            ),
            version.startswith("t4pl_"),
        ]
    ), f"unexpected version: {version}"

    in_dir = osp.join(root_path, version)
    assert osp.exists(in_dir), f"{in_dir} doesn't exist"

    assert dataset_config is not None, "--dataset_config must be specified."

    out_dir = osp.join(out_dir, version)

    with open(dataset_config, "r") as f:
        dataset_config_dict: Dict[str, List[str]] = yaml.safe_load(f)

    t4_infos = {
        "train": [],
        "val": [],
        "test": [],
    }

    name_mapping: Dict[str, str] = get_mapping(version)

    include_camera = not version.startswith("t4pl_")
    cameras = camera_types.T4DATASET if include_camera else []

    do_not_check_valid_flag = version in ["t4xx1_uc2"]

    metainfo = dict(classes=classes.T4DATASET, version=version)

    for split in ["train", "val", "test"]:
        print_log(f"Creating data info for split: {split}", logger="current")
        for scene_id in dataset_config_dict.get(split, []):
            scene_dir = osp.join(root_path, version, scene_id)
            if not osp.isdir(scene_dir):
                raise ValueError(f"{scene_dir} does not exist.")

            nusc = NuScenes(version="annotation", dataroot=scene_dir, verbose=False)

            for i, sample in enumerate(nusc.sample):
                lidar_token = get_lidar_token(sample)
                if lidar_token is None:
                    print_log(
                        f"sample {sample['token']} doesn't have lidar", level=logging.WARNING
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

                info = get_empty_standard_data_info(cameras)

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
                    get_lidar_sweeps_info(nusc, cs_record, pose_record, sd_record, max_sweeps),
                    get_annotations(
                        nusc,
                        sample["anns"],
                        boxes,
                        e2g_r_mat,
                        l2e_r_mat,
                        name_mapping,
                        classes.T4DATASET,
                        do_not_check_valid_flag,
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
        _info_path = osp.join(out_dir, f"{info_prefix}_infos_{_split}.pkl")
        mmengine.dump(dict(data_list=_infos, metainfo=metainfo), _info_path)

    save(t4_infos["train"], "train")
    save(t4_infos["val"], "val")
    save(t4_infos["train"] + t4_infos["val"], "trainval")
    save(t4_infos["test"], "test")
    save(t4_infos["train"] + t4_infos["val"] + t4_infos["test"], "all")


def parse_args():
    parser = argparse.ArgumentParser(description="Data converter")

    parser.add_argument('version', help='name of the dataset')
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--max_sweeps",
        type=int,
        default=10,
        help="[Optional]specify sweeps of lidar per example.(Default: 10)",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="[Optional]output directory of info file. if None, data/<dataset>",
    )
    parser.add_argument(
        "--info_prefix",
        type=str,
        help="[Optional]prefix for info files. if None, <dataset>",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="[Optional]number of threads to be used in waymo conversion.(Default: 4)",
    )
    parser.add_argument(
        "--overwrite_root_path",
        type=str,
        default=None,
        help="specify the root path for sagemaker",
    )
    parser.add_argument(
        "--use_2d_annotation",
        action="store_true",
        help="whether need to include 2d object annotation",
    )
    parser.add_argument(
        "--save_image_in_waymo",
        action="store_true",
        help="whether need to save image in waymo info creation",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="specify the dataset config",
    )
    args = parser.parse_args()
    return args


def process_dataset(
    dataset: str,
    root_path: str,
    version: str,
    max_sweeps: int,
    out_dir: str,
    info_prefix: str = None,
    workers: int = 8,
    overwrite_root_path: str = None,
    use_2d_annotation: bool = False,
    save_image_in_waymo: bool = False,
    dataset_config: str = None,
) -> None:
    info_prefix = dataset if info_prefix is None else info_prefix
    out_dir = os.path.join("data", dataset) if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    t4dataset_data_prep(
        root_path=root_path,
        info_prefix=info_prefix,
        version=version,
        out_dir=out_dir,
        max_sweeps=max_sweeps,
        dataset_config=dataset_config,
        overwrite_root_path=overwrite_root_path,
        use_2d_annotation=use_2d_annotation,
    )


def main():
    args = parse_args()
    process_dataset(
        "t4dataset",
        args.root_path,
        args.version,
        args.max_sweeps,
        args.out_dir,
        args.info_prefix,
        args.workers,
        args.overwrite_root_path,
        args.use_2d_annotation,
        args.save_image_in_waymo,
        args.dataset_config,
    )


if __name__ == "__main__":
    main()
