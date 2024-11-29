import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from mmengine import dump as mmengine_dump
from mmengine.logging import print_log
from mmengine.config import Config
import numpy as np
from nuscenes import NuScenes
from nuscenes.nuscenes import Box

from tools.detection3d.t4dataset_converters.t4converter import (
    extract_nuscenes_data, get_ego2global, get_gt_attrs, get_instances,
    get_lidar_points_info, get_lidar_sweeps_info, obtain_sensor2top,
    match_objects, parse_camera_path)
from tools.detection3d.create_data_t4dataset import get_lidar_token, get_scene_root_dir_path
from tools.detection3d.t4dataset_converters.update_infos_to_v2 import \
    get_empty_standard_data_info

def _create_non_annotated_info(
    cfg: Config,
    dataset_version: str,
    logger: logging.Logger,
    max_sweeps: int = 2,
) -> Path:
    """
    Create non_annotated info file(.pkl)

    Args:
        cfg (Config): Config for the model used for auto labeling.
        dataset_version (str): Version of T4dataset. e.g, "pseudo_xx1"
        logger (logging.Logger): Logger instance for output messages.
        max_sweeps (int, optional): Maximum number of sweeps. Defaults to 2

    Returns:
        Path: Path to non_annotated info file

    Raises:
        ValueError: If scene root directory does not exist
        AssertionError: If no valid samples are found in the dataset
    """
    logger.info(f"Creating data info for split: pseudo")

    ann_file_out_dir = Path(cfg.data_root + cfg.info_directory_path)
    ann_file_out_dir.mkdir(parents=True, exist_ok=True)

    t4_infos: dict[str, list] = {
        "pseudo": [],
    }
    metainfo = dict(classes=cfg.class_names, version=dataset_version)
    dataset_root = Path(cfg.data_root)

    # Get all child directories, excluding hidden directories (starting with '.')
    scene_ids: List[str] = [d.name for d in (Path(cfg.data_root) / dataset_version).iterdir() if d.is_dir() and not d.name.startswith('.')]

    for scene_id in scene_ids:
        scene_root_dir_path = get_scene_root_dir_path(
            dataset_root,
            dataset_version,
            scene_id,
        )

        if not Path(scene_root_dir_path).is_dir():
            raise ValueError(f"{scene_root_dir_path} does not exist.")
        nusc = NuScenes(version="annotation", dataroot=scene_root_dir_path, verbose=False)
        for i, sample in enumerate(nusc.sample):
            info = get_info(cfg, nusc, sample, i, max_sweeps)
            t4_infos["pseudo"].append(info)

    assert len(t4_infos["pseudo"]) > 0, "dataset isn't available"
    logger.info(f"Non annotated sample: {len(t4_infos['pseudo'])}")

    _info_name: str = f"t4dataset_{dataset_version}_infos_pseudo.pkl"
    info_path: Path = ann_file_out_dir / f"t4dataset_{dataset_version}_infos_pseudo.pkl"
    mmengine_dump(dict(data_list=t4_infos["pseudo"], metainfo=metainfo), info_path)

    return info_path

# NOTE: This function is copied from tools.detection3d.t4dataset_converters.t4converter
def get_annotations(
    nusc: NuScenes,
    anns,
    boxes: List[Box],
    e2g_r_mat: np.array,
    l2e_r_mat: np.array,
    name_mapping: dict,
    class_names: List[str],
    filter_attributes: Optional[List[Tuple[str, str]]],
    merge_objects: List[Tuple[str, List[str]]] = [],
    merge_type: str = None,
) -> dict:
    annotations = [nusc.get("sample_annotation", token) for token in anns]
    instance_tokens = [ann["instance_token"] for ann in annotations]
    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0]
                     for b in boxes]).reshape(-1, 1)
    velocity = np.array([nusc.box_velocity(token)[:2] for token in anns])

    valid_flag = np.array([anno["num_lidar_pts"] > 0 for anno in annotations],
                          dtype=bool).reshape(-1)

    # convert velo from global to lidar
    for i in range(len(boxes)):
        velo = np.array([*velocity[i], 0.0])
        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        velocity[i] = velo[:2]

    names = np.array([name_mapping.get(b.name, b.name) for b in boxes])
    # we need to convert rot to SECOND format.
    # Copied from https://github.com/open-mmlab/mmdetection3d/blob/0f9dfa97a35ef87e16b700742d3c358d0ad15452/tools/dataset_converters/nuscenes_converter.py#L258
    gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
    assert len(gt_boxes) == len(
        annotations), f"{len(gt_boxes)}, {len(annotations)}"

    gt_attrs = get_gt_attrs(nusc, annotations)
    assert len(names) == len(gt_attrs), f"{len(names)}, {len(gt_attrs)}"
    assert len(gt_boxes) == len(instance_tokens)
    # assert velocity.shape == (len(gt_boxes), 2)

    matched_object_idx = None
    if merge_objects:
        matched_object_idx = match_objects(gt_boxes, names, merge_objects)

    instances = get_instances(
        gt_boxes,
        names,
        class_names,
        velocity,
        boxes,
        annotations,
        valid_flag,
        gt_attrs,
        filter_attributions=filter_attributes,
        matched_object_idx=matched_object_idx,
        merge_type=merge_type,
    )

    return dict(instances=instances)

# NOTE: This function is copied from tools.detection3d.create_data_t4dataset
def get_info(
    cfg: Any,
    nusc: Any,
    sample: Any,
    i: int,
    max_sweeps: int,
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
            get_lidar_sweeps_info(
                nusc,
                cs_record,
                pose_record,
                sd_record,
                max_sweeps,
            ),
            get_annotations(
                nusc,
                sample["anns"],
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
    if (len(camera_types) > 0):
        for cam in camera_types:
            if cam in sample['data']:
                cam_token = sample['data'][cam]
                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    nusc,
                    cam_token,
                    l2e_t,
                    l2e_r_mat,
                    e2g_t,
                    e2g_r_mat,
                    cam,
                )
                cam_info.update(cam_intrinsic=cam_intrinsic)

                info["images"][cam]['img_path'] = parse_camera_path(
                    cam_info['data_path'])
                info["images"][cam]['cam2img'] = cam_info[
                    'cam_intrinsic'].tolist()
                info["images"][cam]['sample_data_token'] = cam_info[
                    'sample_data_token']
                # bc-breaking: Timestamp has divided 1e6 in pkl infos.
                info["images"][cam]['timestamp'] = cam_info['timestamp'] / 1e6
                info["images"][cam]['cam2ego'] = convert_quaternion_to_matrix(
                    cam_info['sensor2ego_rotation'],
                    cam_info['sensor2ego_translation'])
                lidar2sensor = np.eye(4)
                rot = cam_info['sensor2lidar_rotation']
                trans = cam_info['sensor2lidar_translation']
                lidar2sensor[:3, :3] = rot.T
                lidar2sensor[:3,
                             3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
                info["images"][cam]['lidar2cam'] = lidar2sensor.astype(
                    np.float32).tolist()
                #info["images"].update({cam: cam_info})
    return info
