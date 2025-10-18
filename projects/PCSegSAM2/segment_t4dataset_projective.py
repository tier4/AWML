import argparse
import concurrent.futures
import logging
import os
import os.path as osp
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml
from mmengine.config import Config
from skimage.morphology import binary_dilation, square
from skimage.segmentation import find_boundaries
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

    lidar_e2g_transform = np.eye(4, dtype=np.float32)
    lidar_e2g_transform[0:3, 0:3] = e2g_r_mat
    lidar_e2g_transform[0:3, 3] = e2g_t

    camera_types = cfg.camera_types
    cam_data: List[str, str, np.ndarray, np.ndarray, np.ndarray] = []
    assert len(camera_types) > 0

    projective_segmentation_cfg = segmentation_cfg["projective_segmentation"]
    num_consistent_frames = projective_segmentation_cfg["num_consistent_frames"]

    for cam in camera_types:
        if cam not in sample.data:
            continue

        cam_token = sample.data[cam]

        num_past_frames = num_consistent_frames // 2

        for _ in range(num_past_frames):
            sd_record: SampleData = t4.get("sample_data", cam_token)

            if sd_record.prev != "":
                cam_token = sd_record.prev

        for _ in range(num_consistent_frames):

            sd_record: SampleData = t4.get("sample_data", cam_token)
            cs_record: CalibratedSensor = t4.get("calibrated_sensor", sd_record.calibrated_sensor_token)
            pose_record: EgoPose = t4.get("ego_pose", sd_record.ego_pose_token)

            cam_path, boxes, cam_intrinsics = t4.get_sample_data(cam_token)

            c2e_t = cs_record.translation
            e2g_t = pose_record.translation
            c2e_r = cs_record.rotation
            e2g_r = pose_record.rotation
            c2e_r_mat = c2e_r.rotation_matrix
            e2g_r_mat = e2g_r.rotation_matrix

            c2e_transform = np.eye(4, dtype=np.float32)
            c2e_transform[0:3, 0:3] = c2e_r_mat
            c2e_transform[0:3, 3] = c2e_t

            e2g_transform = np.eye(4, dtype=np.float32)
            e2g_transform[0:3, 0:3] = e2g_r_mat
            e2g_transform[0:3, 3] = e2g_t

            cam2_img_transform = np.eye(4, dtype=np.float32)
            cam2_img_transform[0:3, 0:3] = cam_intrinsics

            cam_data.append(
                [cam, cam_path, np.linalg.inv(e2g_transform), np.linalg.inv(c2e_transform), cam2_img_transform]
            )

            if sd_record.next == "":
                break

            cam_token = sd_record.next

    # Load points
    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
    num_points = points.shape[0]
    points_lcs = np.hstack([points[:, 0:3], np.ones((num_points, 1))])

    points_ecs = points_lcs @ lidar_l2e_transform.T
    points_gcs = points_ecs @ lidar_e2g_transform.T

    # Load segmented images
    seg_pointcloud_list = []
    on_img_mask_list = []

    background_value = projective_segmentation_cfg["background_value"]
    invalid_value = projective_segmentation_cfg["invalid_value"]

    fill_boundaries_with_invalid = projective_segmentation_cfg["fill_boundaries_with_invalid"]
    fill_boundaries_width = projective_segmentation_cfg["fill_boundaries_width"]

    mapping_dict = get_class_mapping(segmentation_cfg)

    for cam, img_path, g2e_transform, e2c_transform, cam2img_transform in cam_data:

        img_path = Path(img_path)
        seg_img_path = img_path.with_name(img_path.stem + "_seg.png")
        seg_image = cv2.imread(str(seg_img_path), cv2.IMREAD_GRAYSCALE).astype(np.int32)
        h, w = seg_image.shape

        # Mapping from SAM2 classes to lidar segmentation classes
        seg_image = np.vectorize(mapping_dict.__getitem__)(seg_image)

        # There are cases where SAM fill the image with one class.
        # Skip those for now
        if seg_image.min() == seg_image.max():
            print(f"class {seg_image.min()} filled the whole image. potential error ({str(img_path)})")
            continue

        if fill_boundaries_with_invalid:

            boundaries = find_boundaries(seg_image, mode="inner", connectivity=1)
            selem = square(fill_boundaries_width)
            boundaries = binary_dilation(boundaries, selem)
            seg_image[boundaries] = background_value

        points_ecs = points_gcs @ g2e_transform.T

        points_ccs = points_ecs @ e2c_transform.T

        points_ics = points_ccs @ cam2img_transform.T
        points_ics[:, 0:2] /= points_ics[:, 2:3]

        on_img_mask = np.logical_and.reduce(
            (
                points_ics[:, 0] > 0,
                points_ics[:, 0] <= w,
                points_ics[:, 1] > 0,
                points_ics[:, 1] <= h,
                points_ics[:, 2] > 0,
            )
        )

        seg_pointcloud = np.full((num_points,), -1, dtype=np.int32)
        seg_pointcloud[on_img_mask] = seg_image[
            points_ics[on_img_mask, 1].astype(np.int32), points_ics[on_img_mask, 0].astype(np.int32)
        ]

        on_img_mask_list.append(on_img_mask)
        seg_pointcloud_list.append(seg_pointcloud)

    # Stack all the segmented points and masks
    seg_pointcloud = np.stack(seg_pointcloud_list, axis=0)
    seg_pointcloud[seg_pointcloud == -1] == invalid_value
    on_img_mask = np.stack(on_img_mask_list, axis=0)

    # Create a masked array
    on_img_non_bg_mask = np.logical_and(on_img_mask, seg_pointcloud != background_value)

    seg_pointcloud_non_bg_masked = np.ma.masked_array(seg_pointcloud, mask=~on_img_non_bg_mask)

    # Check consistency checking differency between min and max
    seg_pointcloud_non_bg_masked_min = np.ma.min(seg_pointcloud_non_bg_masked, axis=0)
    seg_pointcloud_non_bg_masked_max = np.ma.max(seg_pointcloud_non_bg_masked, axis=0)
    seg_pointcloud_non_bg_valid = seg_pointcloud_non_bg_masked_min == seg_pointcloud_non_bg_masked_max
    seg_pointcloud_non_bg_valid.set_fill_value(False)
    seg_pointcloud_non_bg_valid = seg_pointcloud_non_bg_valid.filled()

    seg_pointcloud_combined = np.full((num_points,), invalid_value, dtype=np.uint8)
    seg_pointcloud_combined[seg_pointcloud_non_bg_valid] = seg_pointcloud_non_bg_masked_max[
        seg_pointcloud_non_bg_valid
    ]

    # Dummy ground filter to avoid vehicles and other classes to leak into the ground
    # This may cause small objects to not be classified correctly, but this is just a test
    ground_value = projective_segmentation_cfg["ground_value"]
    min_non_ground_z = projective_segmentation_cfg["min_non_ground_z"]

    points_ecs = points_lcs @ lidar_l2e_transform.T
    update_ground_mask = np.logical_and.reduce(
        (
            seg_pointcloud_combined != invalid_value,
            seg_pointcloud_combined != ground_value,
            points_ecs[:, 2] <= min_non_ground_z,
        )
    )

    seg_pointcloud_combined[update_ground_mask] = invalid_value

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
