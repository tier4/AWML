import argparse
import logging
import os
import os.path as osp
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import cv2
import mmengine
import numpy as np
import yaml
from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from mmengine.config import Config
from mmengine.logging import print_log
from t4_devkit import Tier4
from t4_devkit.common.timestamp import us2sec
from t4_devkit.schema import Sample
from tqdm import tqdm

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
    num_consistent_frames: int,
):
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
            1,
        ),
        get_annotations(
            t4,
            sample.ann_3ds,
            boxes,
            e2g_r_mat,
            l2e_r_mat,
            cfg.name_mapping,
            cfg.class_names,
            cfg.filter_attributes,
            merge_objects=cfg.merge_objects,
            merge_type=cfg.merge_type,
        ),
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

    # Load points
    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
    num_points = points.shape[0]
    points_homog = np.hstack([points[:, 0:3], np.ones((num_points, 1))])

    # Load segmented images
    seg_pointcloud_list = []
    on_img_mask_list = []

    for cam, cam_info in info["images"].items():

        if cam_info["img_path"] is None:
            continue

        img_path = Path(root_path) / cam_info["img_path"]
        seg_img_path = img_path.with_name(img_path.stem + "_seg.png")
        seg_image = cv2.imread(str(seg_img_path), cv2.IMREAD_GRAYSCALE)
        h, w = seg_image.shape

        lidar2cam = np.array(info["images"][cam]["lidar2cam"])
        cam2img = np.eye(4).astype(np.float32)
        cam2img[:3, :3] = np.array(info["images"][cam]["cam2img"])

        lidar2img = cam2img @ lidar2cam

        points_ccs = points_homog @ lidar2cam.T

        points_ics = points_ccs @ cam2img.T
        points_ics[:, 0:2] /= points_ics[:, 2:3]

        on_img_mask = np.logical_and(
            np.logical_and(points_ics[:, 0] > 0, points_ics[:, 0] <= w),
            np.logical_and(points_ics[:, 1] > 0, points_ics[:, 1] <= h),
        )

        seg_pointcloud = np.full((num_points,), -1, dtype=np.int32)
        seg_pointcloud[on_img_mask] = seg_image[
            points_ics[on_img_mask, 1].astype(np.int32), points_ics[on_img_mask, 0].astype(np.int32)
        ]

        on_img_mask_list.append(on_img_mask)
        seg_pointcloud_list.append(seg_pointcloud)

    # Stack all the segmented points and masks
    seg_pointcloud = np.stack(seg_pointcloud_list, axis=0)
    on_img_mask = np.stack(on_img_mask_list, axis=0)

    projective_segmentation_cfg = segmentation_cfg["projective_segmentation"]
    background_value = projective_segmentation_cfg["background_value"]
    invalid_value = projective_segmentation_cfg["invalid_value"]
    mapping_dict = get_class_mapping(segmentation_cfg)

    mapping_dict[-1] = invalid_value
    seg_pointcloud_mapped = np.vectorize(mapping_dict.__getitem__)(seg_pointcloud)

    # Create a masked array
    ong_img_non_bg_mask = np.logical_and(
        on_img_mask, seg_pointcloud_mapped != background_value
    )  # make this a parameter
    ong_img_bg_mask = np.logical_and(on_img_mask, seg_pointcloud_mapped == background_value).max(axis=0)
    seg_pointcloud_non_bg_masked = np.ma.masked_array(seg_pointcloud_mapped, mask=~ong_img_non_bg_mask)

    # Check consistency checking differency between min and max
    seg_pointcloud_non_bg_masked_min = np.ma.min(seg_pointcloud_non_bg_masked, axis=0)
    seg_pointcloud_non_bg_masked_max = np.ma.max(seg_pointcloud_non_bg_masked, axis=0)
    seg_pointcloud_non_bg_valid = seg_pointcloud_non_bg_masked_min == seg_pointcloud_non_bg_masked_max
    seg_pointcloud_non_bg_valid.set_fill_value(False)
    seg_pointcloud_non_bg_valid = seg_pointcloud_non_bg_valid.filled()

    seg_pointcloud_combined = np.full((num_points,), invalid_value, dtype=np.uint8)
    seg_pointcloud_combined[ong_img_bg_mask] = background_value
    seg_pointcloud_combined[seg_pointcloud_non_bg_valid] = seg_pointcloud_non_bg_masked_max[
        seg_pointcloud_non_bg_valid
    ]

    lidar_path = Path(lidar_path)
    basename = lidar_path.name.split(".")[0]
    seg_path = lidar_path.parent / f"{basename}_seg.npy"
    with open(seg_path, "wb") as f:
        np.save(f, seg_pointcloud_combined)

    return


def get_class_mapping(cfg: Any) -> Dict[int, int]:

    sam2_cfg = cfg["sam2"]
    projective_segmentation_cfg = cfg["projective_segmentation"]

    mapping_dict = {}
    mapping_dict[sam2_cfg["background_value"]] = projective_segmentation_cfg["background_value"]

    sam2_class_to_idx = {}
    for i, class_name in enumerate(sam2_cfg["sam2_classes"]):
        sam2_class_to_idx[class_name] = i

    for class_name, segmentation_idx in projective_segmentation_cfg["classes_map"].items():
        mapping_dict[sam2_class_to_idx[class_name]] = segmentation_idx

    return mapping_dict


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
        "--num_consistent_frames",
        type=int,
        required=True,
        help="number of frames that pointclouds need to be consistent",
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

    # TODO(knzo25): hack since I only want to test part of the db
    cfg.dataset_version_list = ["db_jpntaxi_v2"]

    with open(args.segmentation_config, "r") as f:
        segmentation_cfg = yaml.safe_load(f)

    for dataset_version in cfg.dataset_version_list:
        dataset_list = osp.join(cfg.dataset_version_config_root, dataset_version + ".yaml")
        with open(dataset_list, "r") as f:
            dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)

        for split in ["train", "val", "test"]:
            print_log(f"Segmenting split: {split}", logger="current")
            for scene_id in dataset_list_dict.get(split, []):
                print_log(f"Segmenting pointclouds from scene: {scene_id}")
                scene_root_dir_path = get_scene_root_dir_path(
                    args.root_path,
                    dataset_version,
                    scene_id,
                )

                if not osp.isdir(scene_root_dir_path):
                    raise ValueError(f"{scene_root_dir_path} does not exist.")
                t4 = Tier4(version="annotation", data_root=scene_root_dir_path, verbose=False)
                for i, sample in enumerate(tqdm(t4.sample)):
                    segment_pointcloud(
                        args.root_path, cfg, segmentation_cfg, t4, sample, i, args.num_consistent_frames
                    )


if __name__ == "__main__":
    main()
