import argparse
import logging
import multiprocessing as mp
import os
import os.path as osp
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import concurrent.futures

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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


def create_scatter_figure(pointcloud, seg, cmap, min_range, max_range, marker_size=1):

    x_lim = (min_range, max_range)
    y_lim = (min_range, max_range)

    fig, ax = plt.subplots(figsize=(12, 12))

    x = pointcloud[:, 0]
    y = pointcloud[:, 1]

    scatter = ax.scatter(x, y, c=seg, cmap=cmap, s=marker_size)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("BEV Seg")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    return fig, ax


def get_frame_from_fig(fig):

    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, 0:3]

    return img


def generate_bev_segmentation(
    root_path: str,
    cfg: Any,
    segmentation_cfg: Any,
    t4: Tier4,
    sample: Sample,
    cmap: List,
):
    lidar_token = get_lidar_token(sample)
    if lidar_token is None:
        logging.warning(
            f"sample {sample['token']} doesn't have lidar",
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

    lidar_l2e_transform = np.eye(4, dtype=np.float32)
    lidar_l2e_transform[0:3, 0:3] = l2e_r_mat
    lidar_l2e_transform[0:3, 3] = l2e_t

    # Load points
    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
    num_points = points.shape[0]
    points_lcs = np.hstack([points[:, 0:3], np.ones((num_points, 1))])

    points_ecs = points_lcs @ lidar_l2e_transform.T

    lidar_path = Path(lidar_path)
    basename = lidar_path.name.split(".")[0]
    seg_path = lidar_path.parent / f"{basename}_seg.npy"

    seg = np.load(str(seg_path))

    min_range = segmentation_cfg["visualization"]["min_range"]
    max_range = segmentation_cfg["visualization"]["max_range"]

    fig, ax = create_scatter_figure(points_ecs, seg, cmap, min_range, max_range)

    bev_img = get_frame_from_fig(fig)

    plt.close(fig)

    return bev_img


def generate_videos_scene(args, cfg, segmentation_cfg, dataset_version, custom_cmap, scene_id):

    logging.info(f"Creating video for scene: {scene_id}")
    scene_root_dir_path = get_scene_root_dir_path(
        args.root_path,
        dataset_version,
        scene_id,
    )

    if not osp.isdir(scene_root_dir_path):
        raise ValueError(f"{scene_root_dir_path} does not exist.")

    t4 = Tier4(version="annotation", data_root=scene_root_dir_path, verbose=False)

    bev_images = []

    for i, sample in enumerate(tqdm(t4.sample)):
        bev_images.append(generate_bev_segmentation(args.root_path, cfg, segmentation_cfg, t4, sample, custom_cmap))

    generate_video(args.out_videos, scene_id, bev_images)


def generate_videos_scene_wrapper(args):
    return generate_videos_scene(*args)


def parse_args():
    parser = argparse.ArgumentParser(description="Create data info for T4dataset")

    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="config for T4dataset",
    )

    parser.add_argument(
        "--segmentation_config",
        type=str,
        required=True,
        help="segmentation config",
    )

    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="specify the root path of dataset",
    )

    parser.add_argument(
        "--out_videos",
        type=str,
        required=True,
        help="directory to save segmented videos",
    )

    args = parser.parse_args()
    return args


def generate_video(video_folder, scene_id, images):

    if len(images) == 0:
        logging.info("Empty list. Already processed (?)")
        return

    height, width, layers = images[0].shape

    output_file = Path(video_folder) / f"{scene_id}_bev_seg.mp4"
    fps = 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in images:
        image = cv2.resize(image, (width, height))
        video_writer.write(image)

    video_writer.release()
    logging.info(f"Video created successfully: {output_file}")


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    # load config
    cfg = Config.fromfile(args.dataset_config)
    os.makedirs(args.out_videos, exist_ok=True)

    with open(args.segmentation_config, "r") as f:
        segmentation_cfg = yaml.safe_load(f)

    # TODO(knzo25): hack since I only want to test part of the db
    cfg.dataset_version_list = ["db_jpntaxi_v2"]

    # Create cmap
    cmap_dict = segmentation_cfg["visualization"]["color_map"]
    cmap_list = [cmap_dict[i] if i in cmap_dict else [0.0, 0.0, 0.0] for i in range(0, 256)]
    custom_cmap = mcolors.ListedColormap(cmap_list)

    num_workers = segmentation_cfg["projective_segmentation"]["num_workers"]

    for dataset_version in cfg.dataset_version_list:
        dataset_list = osp.join(cfg.dataset_version_config_root, dataset_version + ".yaml")

        with open(dataset_list, "r") as f:
            dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)

        for split in ["train", "val", "test"]:
            logging.info(f"Creating videos from split: {split}")

            scenes_list = dataset_list_dict.get(split, [])
            pool_args = [
                (args, cfg, segmentation_cfg, dataset_version, custom_cmap, scene_id) for scene_id in scenes_list
            ]

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                executor.map(generate_videos_scene_wrapper, pool_args)


if __name__ == "__main__":
    main()
