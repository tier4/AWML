import logging
from pathlib import Path
from pyquaternion import Quaternion
from typing import Any, Dict, List, Optional, Tuple

from mmdet3d.datasets.utils import convert_quaternion_to_matrix
import mmengine
from mmengine import dump as mmengine_dump
from mmengine.logging import print_log
from mmengine.config import Config
import numpy as np
from t4_devkit import Tier4
from t4_devkit.dataclass import Box3D
from t4_devkit.schema import Sample, SampleAnnotation, CalibratedSensor, SampleData, EgoPose
from t4_devkit.common.timestamp import us2sec

from tools.detection3d.t4dataset_converters.t4converter import (
    extract_tier4_data, get_ego2global, get_gt_attrs, get_instances,
    obtain_sensor2top, match_objects)
from tools.detection3d.create_data_t4dataset import get_lidar_token, get_scene_root_dir_path
from tools.detection3d.t4dataset_converters.update_infos_to_v2 import (
    get_empty_standard_data_info, get_single_lidar_sweep)


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
    scene_ids: List[str] = [
        d.name for d in (Path(cfg.data_root) / dataset_version).iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]

    for scene_id in scene_ids:
        scene_root_dir_path = get_scene_root_dir_path(
            dataset_root,
            dataset_version,
            scene_id,
        )

        if not Path(scene_root_dir_path).is_dir():
            raise ValueError(f"{scene_root_dir_path} does not exist.")
        t4 = Tier4(
            version="annotation", data_root=scene_root_dir_path, verbose=False)
        for i, sample in enumerate(t4.sample):
            info = get_info(cfg, t4, sample, i, max_sweeps)
            t4_infos["pseudo"].append(info)

    assert len(t4_infos["pseudo"]) > 0, "dataset isn't available"
    logger.info(f"Non annotated sample: {len(t4_infos['pseudo'])}")

    _info_name: str = f"t4dataset_{dataset_version}_infos_pseudo.pkl"
    info_path: Path = ann_file_out_dir / f"t4dataset_{dataset_version}_infos_pseudo.pkl"
    mmengine_dump(
        dict(data_list=t4_infos["pseudo"], metainfo=metainfo), info_path)

    return info_path

# NOTE: This function is copied from tools.detection3d.t4dataset_converters.t4converter and modified for non-annotated t4. non-annotated t4 does not have dataset-version.
def parse_camera_path(camera_path: str) -> str:
    """leave only {database_version}/{scene_id}/data/{camera_type}/{frame}.jpg from path"""
    return "/".join(camera_path.split("/")[-5:])

# NOTE: This function is copied from tools.detection3d.t4dataset_converters.t4converter and modified for non-annotated t4. non-annotated t4 does not have dataset-version.
def parse_lidar_path(lidar_path: str) -> str:
    """leave only {database_version}/{scene_id}/data/{lidar_token}/{frame}.bin from path"""
    return "/".join(lidar_path.split("/")[-5:])

# NOTE: This function is copied from tools.detection3d.t4dataset_converters.t4converter
def get_lidar_points_info(
    lidar_path: str,
    cs_record: CalibratedSensor,
    num_features: int = 5,
):
    lidar2ego = convert_quaternion_to_matrix(
        quaternion=cs_record.rotation, translation=cs_record.translation)
    mmengine.check_file_exist(lidar_path)
    lidar_path = parse_lidar_path(lidar_path)
    return dict(
        lidar_points=dict(
            num_pts_feats=num_features,
            lidar_path=lidar_path,
            lidar2ego=lidar2ego,
        ), )

# NOTE: This function is copied from tools.detection3d.t4dataset_converters.t4converter
def get_lidar_sweeps_info(
    t4: Tier4,
    cs_record: CalibratedSensor,
    pose_record: EgoPose,
    sd_rec: SampleData,
    max_sweeps: int,
    num_features: int = 5,
):
    l2e_r = cs_record.rotation
    l2e_t = cs_record.translation
    e2g_r = pose_record.rotation
    e2g_t = pose_record.translation
    l2e_r_mat = l2e_r.rotation_matrix
    e2g_r_mat = e2g_r.rotation_matrix

    sweeps = []
    while len(sweeps) < max_sweeps:
        if not sd_rec.prev == "":
            sweep = get_single_lidar_sweep()
            v1_sweep = obtain_sensor2top(
                t4,
                sd_rec.prev,
                l2e_t,
                l2e_r_mat,
                e2g_t,
                e2g_r_mat,
                "lidar",
            )

            sweep["timestamp"] = v1_sweep["timestamp"] / 1e6
            sweep["sample_data_token"] = v1_sweep["sample_data_token"]
            sweep["ego2global"] = convert_quaternion_to_matrix(
                quaternion=v1_sweep["ego2global_rotation"],
                translation=v1_sweep["ego2global_translation"],
            )

            rot = v1_sweep["sensor2lidar_rotation"]
            trans = v1_sweep["sensor2lidar_translation"]
            lidar2sensor = np.eye(4)
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))

            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = rot
            lidar2ego[:3, 3] = trans
            lidar_path = v1_sweep["data_path"]

            mmengine.check_file_exist(lidar_path)
            lidar_path = parse_lidar_path(lidar_path)

            sweep["lidar_points"] = dict(
                lidar_path=lidar_path,
                lidar2ego=lidar2ego,
                num_pts_feats=num_features,
                lidar2sensor=lidar2sensor.astype(np.float32).tolist(),
            )

            sweeps.append(sweep)
            sd_rec = t4.get("sample_data", sd_rec.prev)
        else:
            break

    return dict(lidar_sweeps=sweeps)

# NOTE: This function is copied from tools.detection3d.t4dataset_converters.t4converter
def get_annotations(
    t4: Tier4,
    anns: list[str],
    boxes: List[Box3D],
    e2g_r_mat: np.array,
    l2e_r_mat: np.array,
    name_mapping: dict,
    class_names: List[str],
    filter_attributes: Optional[List[Tuple[str, str]]],
    merge_objects: List[Tuple[str, List[str]]] = [],
    merge_type: str = None,
) -> dict:
    annotations: list[SampleAnnotation] = [
        t4.get("sample_annotation", token) for token in anns
    ]
    instance_tokens = [ann.instance_token for ann in annotations]
    locs = np.array([b.position for b in boxes]).reshape(-1, 3)
    dims = np.array([b.size for b in boxes]).reshape(-1, 3)
    rots = np.array([b.rotation.yaw_pitch_roll[0]
                     for b in boxes]).reshape(-1, 1)
    velocity = np.array([t4.box_velocity(token)[:2] for token in anns])

    valid_flag = np.array([ann.num_lidar_pts > 0 for ann in annotations],
                          dtype=bool).reshape(-1)

    # convert velo from global to lidar
    for i in range(len(boxes)):
        velo = np.array([*velocity[i], 0.0])
        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        velocity[i] = velo[:2]

    names = np.array([
        name_mapping.get(b.semantic_label.name, b.semantic_label.name)
        for b in boxes
    ])
    # we need to convert rot to SECOND format.
    # Copied from https://github.com/open-mmlab/mmdetection3d/blob/0f9dfa97a35ef87e16b700742d3c358d0ad15452/tools/dataset_converters/nuscenes_converter.py#L258
    gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
    assert len(gt_boxes) == len(
        annotations), f"{len(gt_boxes)}, {len(annotations)}"

    gt_attrs = get_gt_attrs(t4, annotations)
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
    t4: Tier4,
    sample: Sample,
    i: int,
    max_sweeps: int,
):
    lidar_token = get_lidar_token(sample)
    if lidar_token is None:
        print_log(
            f"sample {sample.token} doesn't have lidar",
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
                max_sweeps,
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
    if (len(camera_types) > 0):
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
