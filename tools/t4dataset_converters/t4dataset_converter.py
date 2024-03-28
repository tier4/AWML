from enum import Enum
import os
from typing import Dict

import numpy as np
from pyquaternion import Quaternion

from autoware_ml.logger.logger import configure_logger

logger = configure_logger(modname=__name__)


class SplitType(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3

def get_lidar_token(sample_rec: Dict[str, Dict[str, str]]) -> str:
    data_dict = sample_rec["data"]
    if "LIDAR_TOP" in data_dict:
        return data_dict["LIDAR_TOP"]
    elif "LIDAR_CONCAT" in data_dict:
        return data_dict["LIDAR_CONCAT"]
    else:
        return None

def _obtain_sensor2top(
    nusc,
    sensor_token,
    l2e_t,
    l2e_r_mat,
    e2g_t,
    e2g_r_mat,
    sensor_type="lidar",
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
    sweep = _obtain_sensor2top(
        nusc,
        sensor_token,
        l2e_t,
        l2e_r_mat,
        e2g_t,
        e2g_r_mat,
        sensor_type,
    )

    # add annotations
    if load_boxes_3d_in_sweeps and sensor_type == "lidar":
        # NOTE(kan-bayashi): When the frame corresponding to the `sensor_token` is not key-frame,
        #   nusc will return the linear interpolated boxes using boxes of current and prev
        #   key-frames. If the object does not exist in prev key-frame, the box in current
        #   key-frame will be used. Therefore, the instance tokens of `sensor_token` must be the
        #   same the current key-frame's instance tokens.
        #   https://github.com/nutonomy/nuscenes-devkit/blob/da3c9a977112fca05413dab4e944d911769385a9/python-sdk/nuscenes/nuscenes.py#L319-L375
        sd_rec = nusc.get("sample_data", sensor_token)
        _, boxes, _ = nusc.get_sample_data(sensor_token)
        ann_tokens = nusc.get("sample", sd_rec["sample_token"])["anns"]
        anns = [nusc.get("sample_annotation", ann_token) for ann_token in ann_tokens]
        instance_tokens = [ann["instance_token"] for ann in anns]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        velocity = np.array(
            [nusc.box_velocity(ann_token)[:2] for ann_token in ann_tokens]
        ).reshape(-1, 2)
        # convert velo from global to lidar
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]

        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2, velocity], axis=1)
        sweep["gt_boxes_with_velocity"] = gt_boxes
        sweep["instance_tokens"] = instance_tokens

    return sweep

def _obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
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
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep
