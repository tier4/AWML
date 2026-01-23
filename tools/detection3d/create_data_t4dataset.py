import argparse
import logging
import os
import os.path as osp
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mmengine
import numpy as np
import yaml
from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from mmengine.config import Config
from mmengine.logging import print_log
from t4_devkit import Tier4
from t4_devkit.common.timestamp import us2sec
from t4_devkit.schema import Sample, SampleData

from tools.detection3d.t4dataset_converters.t4converter import (
    extract_tier4_data,
    get_annotations,
    get_ego2global,
    get_lidar_points_info,
    get_lidar_sweeps_info,
    get_lidarseg_annotations,
    obtain_sensor2top,
    parse_camera_path,
)
from tools.detection3d.t4dataset_converters.t4dataset_statistics import T4DatasetSceneMetadata, T4DatasetStatistics
from tools.detection3d.t4dataset_converters.update_infos_to_v2 import get_empty_standard_data_info

_UNKNOWN = "unknown"


def _get_bucket_name(
    city: Optional[str], vehicle_type: Optional[str], range_filter_name: str, bev_distance_range: Tuple[float, float]
) -> str:
    """Get a bucket name given city/vehicle_type."""
    if city is None:
        city = _UNKNOWN
    if vehicle_type is None:
        vehicle_type = _UNKNOWN
    return f"{city}/{vehicle_type}/{range_filter_name}_{bev_distance_range[0]}-{bev_distance_range[1]}"


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

    # "./data/t4dataset/db_jpntaxi_v2/e6d0237c-274c-4872-acc9-dc7ea2b77943"
    scene_root_dir_path = osp.join(root_path, dataset_version, scene_id)

    version_dirs = [d for d in os.listdir(scene_root_dir_path) if version_pattern.match(d)]

    if version_dirs:
        version_id = sorted(version_dirs, key=int)[-1]
        # "./data/t4dataset/db_jpntaxi_v2/e6d0237c-274c-4872-acc9-dc7ea2b77943/0"
        return os.path.join(scene_root_dir_path, version_id)
    else:
        warnings.simplefilter("always")
        warnings.warn(
            f"The directory structure of T4 Dataset is deprecated. In the newer version, the directory structure should look something like `$T4DATASET_ID/$VERSION_ID/`. Please update your Web.Auto CLI to the latest version.",
            DeprecationWarning,
        )
        return scene_root_dir_path


def get_info(
    cfg: Any,
    t4: Tier4,
    sample: Sample,
    i: int,
    max_sweeps: int,
    city: Optional[str] = None,
    vehicle_type: Optional[str] = None,
) -> Dict[str, Any]:
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
    ) = extract_tier4_data(t4, sample, lidar_token)

    sd_record: SampleData = t4.get("sample_data", lidar_token)

    info = get_empty_standard_data_info(cfg.camera_types)

    basic_info = dict(
        sample_idx=i,
        token=sample.token,
        timestamp=us2sec(sample.timestamp),
        scene_token=sample.scene_token,
        location=log_record.location,
        scene_name=scene_record.name,
        city=city,
        vehicle_type=vehicle_type,
    )

    for new_info in [
        basic_info,
        get_ego2global(pose_record),
        get_lidar_points_info(lidar_path, cs_record),
        get_lidar_sweeps_info(
            t4,
            cs_record,
            pose_record,
            sd_record,
            max_sweeps,
        ),
        get_annotations(
            t4,
            sample.ann_3ds,
            boxes,
            e2g_r_mat,
            l2e_r_mat,
            cfg,
        ),
        get_lidarseg_annotations(t4, sd_record, i, lidar_token),
    ]:
        info.update(new_info)

    camera_types = cfg.camera_types
    if len(camera_types) > 0:
        for cam in camera_types:
            if cam in sample.data:
                cam_token = sample.data[cam]
                _, _, cam_intrinsic = t4.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    t4,
                    cam_token,
                    l2e_t,
                    l2e_r_mat,
                    e2g_t,
                    e2g_r_mat,
                    cam,
                )
                cam_info.update(cam_intrinsic=cam_intrinsic)

                info["images"][cam]["img_path"] = parse_camera_path(cam_info["data_path"])
                info["images"][cam]["cam2img"] = cam_info["cam_intrinsic"].tolist()
                info["images"][cam]["sample_data_token"] = cam_info["sample_data_token"]
                # bc-breaking: Timestamp has divided 1e6 in pkl infos.
                info["images"][cam]["timestamp"] = cam_info["timestamp"] / 1e6
                info["images"][cam]["cam2ego"] = convert_quaternion_to_matrix(
                    cam_info["sensor2ego_rotation"], cam_info["sensor2ego_translation"]
                )
                lidar2sensor = np.eye(4)
                rot = cam_info["sensor2lidar_rotation"]
                trans = cam_info["sensor2lidar_translation"]
                lidar2sensor[:3, :3] = rot.T
                lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
                info["images"][cam]["lidar2cam"] = lidar2sensor.astype(np.float32).tolist()
                # info["images"].update({cam: cam_info})
    return info


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
    parser.add_argument(
        "--dataset_version_config_root",
        type=str,
        default="autoware_ml/configs/t4dataset/",
        help="specify the root path for yaml t4dataset split",
    )
    parser.add_argument(
        "--use_available_dataset_version",
        action="store_true",
        help="Will resort to using the available dataset version if the one specified in the config file does not exist.",
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

    if cfg.merge_objects:
        for target, sub_objects in cfg.merge_objects:
            assert len(sub_objects) == 2, "Only merging two objects in supported at the moment"

    if cfg.filter_attributes is None:
        print_log("No attribute filtering is applied!")

    # Get every pair of min-max distance filtering thresholds
    min_distance = cfg.evaluator_metric_configs["min_distance"]
    max_distance = cfg.evaluator_metric_configs["max_distance"]

    # TODO(KokSeang): make this configurable, and consistent with evaluation config
    range_filter_name = "bev_center"
    bev_distance_ranges = []
    for min_dist, max_dist in zip(min_distance, max_distance):
        bev_distance_ranges.append((min_dist, max_dist))

    # Generate statistics for this split
    t4_statistics = {
        "train": T4DatasetStatistics(Path(args.out_dir), "train", args.version, cfg.class_names),
        "val": T4DatasetStatistics(Path(args.out_dir), "val", args.version, cfg.class_names),
        "test": T4DatasetStatistics(Path(args.out_dir), "test", args.version, cfg.class_names),
    }
    for dataset_version in cfg.dataset_version_list:
        dataset_list = osp.join(args.dataset_version_config_root, dataset_version + ".yaml")
        with open(dataset_list, "r") as f:
            dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)

        for split in ["train", "val", "test"]:
            print_log(f"Creating data info for split: {split}", logger="current")
            # Get 2 Hz
            if split == "train" and dataset_version == "db_jpntaxigen2_v1":
                sample_steps = 5
            else:
                sample_steps = 1

            for scene_id in dataset_list_dict.get(split, []):
                print_log(f"Creating data info for scene: {scene_id}, steps: {sample_steps}")
                dataset_scene_info = scene_id.split("/")
                if len(dataset_scene_info) == 4:
                    t4_dataset_id, t4_dataset_version_id, city, vehicle_type = dataset_scene_info
                elif len(dataset_scene_info) == 2:
                    t4_dataset_id, t4_dataset_version_id = dataset_scene_info
                    city = vehicle_type = None
                else:
                    raise ValueError(
                        "Invalid scene_id format. should be : {t4_dataset_id}/{t4_dataset_version_id}/{city:optional}/{vehicle_type:optional}"
                    )

                scene_root_dir_path = osp.join(args.root_path, dataset_version, t4_dataset_id, t4_dataset_version_id)
                if not os.path.exists(scene_root_dir_path):
                    if args.use_available_dataset_version:
                        print(
                            "Warning: The version of the dataset specified in the config file does not exist. Will use whatever is available locally."
                        )
                        scene_root_dir_path = get_scene_root_dir_path(args.root_path, dataset_version, t4_dataset_id)
                    else:
                        raise ValueError(f"{t4_dataset_id} does not exist.")
                t4 = Tier4(data_root=scene_root_dir_path, verbose=False)
                info = {}
                for i in range(0, len(t4.sample), sample_steps):
                    sample = t4.sample[i]
                    info = get_info(cfg, t4, sample, i, args.max_sweeps, city, vehicle_type)
                    # info["version"] = dataset_version             # used for visualizations during debugging.
                    t4_infos[split].append(info)

                scene_metadata = T4DatasetSceneMetadata(scene_id, city, vehicle_type)
                for bev_distance_range in bev_distance_ranges:
                    bucket_name = _get_bucket_name(city, vehicle_type, range_filter_name, bev_distance_range)
                    t4_statistics[split].add_samples(t4.sample, info, bucket_name, scene_metadata, bev_distance_range)

                    # Add version statistics without city/vehicle_type
                    bucket_name = _get_bucket_name(args.version, args.version, range_filter_name, bev_distance_range)
                    t4_statistics[split].add_samples(t4.sample, info, bucket_name, scene_metadata, bev_distance_range)

    for t4_statistic_info in t4_statistics.values():
        t4_statistic_info.save_to_parquet()
        print_log(f"Saved {t4_statistic_info.split_name} statistics to {t4_statistic_info.output_dir}")

    assert sum(len(split) for split in t4_infos.values()) > 0, "dataset isn't available"
    print(
        f"train sample: {len(t4_infos['train'])}, "
        f"val sample: {len(t4_infos['val'])}, "
        f"test sample: {len(t4_infos['test'])}"
    )

    def save(_infos, _split):
        _info_path = osp.join(args.out_dir, f"t4dataset_{args.version}_infos_{_split}.pkl")
        mmengine.dump(dict(data_list=_infos, metainfo=metainfo), _info_path)

    save(t4_infos["train"], "train")
    save(t4_infos["val"], "val")
    save(t4_infos["train"] + t4_infos["val"], "trainval")
    save(t4_infos["test"], "test")
    save(t4_infos["train"] + t4_infos["val"] + t4_infos["test"], "all")


if __name__ == "__main__":
    main()
