import copy
import os
from typing import Dict, List

import mmengine
import numpy as np
from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from nuscenes import NuScenes
from nuscenes.nuscenes import Box
from pyquaternion import Quaternion

from tools.detection3d.t4dataset_converters.update_infos_to_v2 import (
    clear_instance_unused_keys, get_empty_instance, get_single_lidar_sweep)


def get_ego2global(pose_record: Dict) -> Dict[str, List]:
    ego2global = convert_quaternion_to_matrix(
        quaternion=pose_record["rotation"],
        translation=pose_record["translation"])
    return dict(ego2global=ego2global, )


def parse_lidar_path(lidar_path: str) -> str:
    """leave only {database_version}/{scene_id}/{dataset_version}/data/{lidar_token}/{frame}.bin from path"""
    return "/".join(lidar_path.split("/")[-6:])


def get_lidar_points_info(
    lidar_path: str,
    cs_record: Dict,
    num_features: int = 5,
):
    lidar2ego = convert_quaternion_to_matrix(
        quaternion=cs_record["rotation"], translation=cs_record["translation"])
    mmengine.check_file_exist(lidar_path)
    lidar_path = parse_lidar_path(lidar_path)
    return dict(
        lidar_points=dict(
            num_pts_feats=num_features,
            lidar_path=lidar_path,
            lidar2ego=lidar2ego,
        ), )


def get_lidar_sweeps_info(
    nusc: NuScenes,
    cs_record: Dict,
    pose_record: Dict,
    sd_rec: Dict,
    max_sweeps: int,
    num_features: int = 5,
):
    l2e_r = cs_record["rotation"]
    l2e_t = cs_record["translation"]
    e2g_r = pose_record["rotation"]
    e2g_t = pose_record["translation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    sweeps = []
    while len(sweeps) < max_sweeps:
        if not sd_rec["prev"] == "":
            sweep = get_single_lidar_sweep()
            v1_sweep = obtain_sensor2top(
                nusc,
                sd_rec["prev"],
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
            sd_rec = nusc.get("sample_data", sd_rec["prev"])
        else:
            break

    return dict(lidar_sweeps=sweeps)


def extract_nuscenes_data(nusc: NuScenes, sample, lidar_token: str):
    sd_record = nusc.get("sample_data", lidar_token)
    cs_record = nusc.get("calibrated_sensor",
                         sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
    mmengine.check_file_exist(lidar_path)

    scene_record = nusc.get("scene", sample["scene_token"])
    log_record = nusc.get("log", scene_record["log_token"])

    l2e_r = cs_record["rotation"]
    e2g_r = pose_record["rotation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
    return (
        pose_record,
        cs_record,
        sd_record,
        scene_record,
        log_record,
        boxes,
        lidar_path,
        e2g_r_mat,
        l2e_r_mat,
    )


def get_gt_attrs(nusc, annotations, attr_categories_mapper) -> List:
    gt_attrs = []
    for anno in annotations:
        if len(anno["attribute_tokens"]) == 0:
            gt_attrs.append("none")
        else:
            attr_names = [
                nusc.get("attribute", t)["name"]
                for t in anno["attribute_tokens"]
            ]
            attr_categories = [a.split(".")[0] for a in attr_names]
            if attr_categories_mapper("pedestrian_state") in attr_categories:
                gt_attrs.append(attr_names[attr_categories.index(
                    attr_categories_mapper("pedestrian_state"))])
            elif attr_categories_mapper("cycle_state") in attr_categories:
                gt_attrs.append(attr_names[attr_categories.index(
                    attr_categories_mapper("cycle_state"))])
            elif attr_categories_mapper("vehicle_state") in attr_categories:
                gt_attrs.append(attr_names[attr_categories.index(
                    attr_categories_mapper("vehicle_state"))])
            elif attr_categories_mapper("occlusion_state") in attr_categories:
                gt_attrs.append(attr_names[attr_categories.index(
                    attr_categories_mapper("occlusion_state"))])
            else:
                raise ValueError(f"invalid attributes: {attr_names}")
    return gt_attrs


def get_instances(
    gt_boxes,
    names,
    class_names,
    velocity,
    boxes,
    annotations,
    valid_flag,
    gt_attrs,
    filter_attributions=[["vehicle.bicycle", "vehicle_state.parked"],
                         ["vehicle.motorcycle", "vehicle_state.parked"]]):
    instances = []
    ignore_class_name = set()
    for i, box in enumerate(gt_boxes):
        empty_instance = get_empty_instance()
        empty_instance["bbox_3d"] = box.tolist()
        if names[i] in class_names:
            is_filter = False
            if filter_attributions:
                for filter_attribution in filter_attributions:
                    if boxes[i].name == filter_attribution[0] and gt_attrs[
                            i] == filter_attribution[1]:
                        is_filter = True
            if is_filter is True:
                empty_instance["bbox_label"] = -1
            else:
                empty_instance["bbox_label"] = class_names.index(names[i])

        else:
            ignore_class_name.add(names[i])
            empty_instance["bbox_label"] = -1
        empty_instance["bbox_label_3d"] = copy.deepcopy(
            empty_instance["bbox_label"])

        empty_instance["velocity"] = velocity.reshape(-1, 2)[i].tolist()
        empty_instance["num_lidar_pts"] = annotations[i]["num_lidar_pts"]
        empty_instance["num_radar_pts"] = annotations[i]["num_radar_pts"]
        empty_instance["bbox_3d_isvalid"] = valid_flag[i]
        empty_instance["gt_nusc_name"] = boxes[i].name
        empty_instance["gt_attrs"] = gt_attrs[i]
        empty_instance = clear_instance_unused_keys(empty_instance)

        instances.append(empty_instance)
    return instances


def get_annotations(
    nusc: NuScenes,
    anns,
    boxes: List[Box],
    e2g_r_mat: np.array,
    l2e_r_mat: np.array,
    name_mapping: dict,
    class_names: List[str],
    attr_categories_mapper=lambda x: x,
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

    gt_attrs = get_gt_attrs(nusc, annotations, attr_categories_mapper)

    assert len(names) == len(gt_attrs), f"{len(names)}, {len(gt_attrs)}"
    assert len(gt_boxes) == len(instance_tokens)

    instances = get_instances(gt_boxes, names, class_names, velocity, boxes,
                              annotations, valid_flag, gt_attrs)

    return dict(instances=instances)


def obtain_sensor2top(
    nusc,
    sensor_token,
    l2e_t,
    l2e_r_mat,
    e2g_t,
    e2g_r_mat,
    sensor_type="lidar",
    root_path=None,
    overwrite_root_path=None,
    name_mapping=None,
    load_boxes_3d_in_sweeps=True,
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.
        root_path (str): Path of the root data. Default: None.
        overwrite_root_path (str): If not None, replace root_path of lidar_path with this argument.
        name_mapping (dict): Mapping from nuScenes category name to target class name.
        load_boxes_3d_in_sweeps (bool): Whether to load 3D boxes in sweeps.

    Returns:
        sweep (dict): Sweep information after transformation.

    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor",
                         sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    if root_path is not None and overwrite_root_path is not None:
        if data_path.startswith("/"):
            data_path = data_path.replace(root_path, overwrite_root_path)
        else:
            data_path = os.path.join(overwrite_root_path, data_path)
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) +
        l2e_t @ np.linalg.inv(l2e_r_mat).T)
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T

    # add annotations
    if load_boxes_3d_in_sweeps and sensor_type == "lidar":
        # NOTE(kan-bayashi): When the frame corresponding to the `sensor_token` is not key-frame,
        #   nusc will return the linear interpolated boxes using boxes of current and prev
        #   key-frames. If the object does not exist in prev key-frame, the box in current
        #   key-frame will be used. Therefore, the instance tokens of `sensor_token` must be the
        #   same the current key-frame's instance tokens.
        #   https://github.com/nutonomy/nuscenes-devkit/blob/da3c9a977112fca05413dab4e944d911769385a9/python-sdk/nuscenes/nuscenes.py#L319-L375
        _, boxes, _ = nusc.get_sample_data(sensor_token)
        ann_tokens = nusc.get("sample", sd_rec["sample_token"])["anns"]
        anns = [
            nusc.get("sample_annotation", ann_token)
            for ann_token in ann_tokens
        ]
        instance_tokens = [ann["instance_token"] for ann in anns]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        velocity = np.array([
            nusc.box_velocity(ann_token)[:2] for ann_token in ann_tokens
        ]).reshape(-1, 2)
        # convert velo from global to lidar
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T
            velocity[i] = velo[:2]

        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2, velocity],
                                  axis=1)
        sweep["gt_boxes_with_velocity"] = gt_boxes
        sweep["instance_tokens"] = instance_tokens

    return sweep
