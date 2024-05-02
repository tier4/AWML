import os
import re
import warnings
from enum import Enum
from typing import Dict, Optional

warnings.simplefilter("always")

import mmengine
import numpy as np
from nuimages import NuImages
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from autoware_ml.detection3d.datasets import t4x2_dataset, t4xx1_dataset


class SplitType(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3


def search_version_if_exists(root_path: str) -> str:
    """
    Search the version of the T4 Dataset if it exists.
    If the version directory under `root_path` exists, for instance as `{root_path}/0`, return the version directory.
    Otherwise, return the root path itself.

    Args:
        root_path (str): The root path of the T4 Dataset.

    Returns:
        str: The version directory if it exists, otherwise the root path itself.
    """
    version_pattern = re.compile(r"^\d+$")  # an integer larger than or equal to 0
    base_dir = os.path.basename(root_path)
    if not version_pattern.match(base_dir):
        version_dirs = [d for d in os.listdir(root_path) if version_pattern.match(d)]
        if version_dirs:
            version_id = sorted(version_dirs, key=int)[-1]
            nusc_root_path = os.path.join(root_path, version_id)
        else:
            warnings.warn(
                f"The directory structure of T4 Dataset is deprecated. In the newer version, the directory structure should look something like `$T4DATASET_ID/$VERSION_ID/`. Please update your Web.Auto CLI to the latest version.",
                DeprecationWarning,
            )
            nusc_root_path = root_path
    else:
        nusc_root_path = root_path
    return nusc_root_path


def create_t4dataset_infos_by_split_type(
    root_path: str,
    version: str,
    split_type: SplitType,
    max_sweeps: int,
    use_2d_annotation: bool,
    overwrite_root_path: Optional[str] = None,
):
    """Create info file of t4dataset by split type.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        version (str): Version of the data.
        dataset_id (str): ID of the t4dataset.
        split_type (SplitType): The type of split. train, val, test.
        max_sweeps (int): Max number of sweeps.
        use_2d_annotation (bool): Whether to use 2d annotation.
        overwrite_root_path (str): If not None, replace root_path of lidar_path with this argument.

    """
    nusc_root_path = search_version_if_exists(root_path)
    nusc = NuScenes(version="annotation", dataroot=nusc_root_path, verbose=False)

    if version in ["t4xx1", "t4xx1_uc2"]:
        name_mapping = t4xx1_dataset.T4XX1Dataset.NameMapping
    elif version == "t4x2":
        name_mapping = t4x2_dataset.T4X2Dataset.NameMapping
    else:
        raise ValueError(f"not supported version: {version}")

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scenes_token = [s["token"] for s in available_scenes]

    train_scenes = set()
    val_scenes = set()
    test_scenes = set()
    if split_type == SplitType.TRAIN:
        train_scenes = set(available_scenes_token)
        val_scenes = set()
        test_scenes = set()
    elif split_type == SplitType.VAL:
        train_scenes = set()
        val_scenes = set(available_scenes_token)
        test_scenes = set()
    elif split_type == SplitType.TEST:
        train_scenes = set()
        val_scenes = set()
        test_scenes = set(available_scenes_token)
    # print(
    #    "train scene: {}, val scene: {}, test scene: {}".format(
    #        len(train_scenes), len(val_scenes), len(test_scenes)
    #    )
    # )

    if len(train_scenes) + len(val_scenes) + len(test_scenes) == 0:
        return [], [], []

    if use_2d_annotation:
        obj_file_path = os.path.join(root_path, "annotation", "object_ann.json")
        if not os.path.exists(obj_file_path):
            raise ValueError("file not exist: {}".format(obj_file_path))
        nuim = NuImages(
            dataroot=root_path, version="annotation", verbose=False
        )  # for loading 2d annos
    else:
        nuim = None

    train_infos, val_infos, test_infos = _fill_infos(
        nusc,
        nuim,
        train_scenes,
        val_scenes,
        test_scenes,
        name_mapping,
        test=False,
        max_sweeps=max_sweeps,
        include_camera=use_2d_annotation or not version.startswith("t4pl_"),
        root_path=root_path.split(version)[0][:-1],
        overwrite_root_path=overwrite_root_path,
        do_not_check_valid_flag=version in ["t4xx1_uc2"],
        load_boxes_3d_in_sweeps=version in ["t4xx1", "t4x2", "t4x2_awsim"]
        or version.startswith("t4pl_"),
    )

    return train_infos, val_infos, test_infos


def get_lidar_token(sample_rec: Dict[str, Dict[str, str]]) -> str:
    data_dict = sample_rec["data"]
    if "LIDAR_TOP" in data_dict:
        return data_dict["LIDAR_TOP"]
    elif "LIDAR_CONCAT" in data_dict:
        return data_dict["LIDAR_CONCAT"]
    else:
        return None


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the available scenes.

    """
    available_scenes = []
    # print("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        lidar_token = get_lidar_token(sample_rec)
        if lidar_token is None:
            continue
        sd_rec = nusc.get("sample_data", lidar_token)
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    # print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


def _fill_infos(
    nusc: NuScenes,
    nuim: Optional[NuImages],
    train_scenes,
    val_scenes,
    test_scenes,
    name_mapping,
    test=False,
    max_sweeps=10,
    attr_categories_mapper=lambda x: x,
    include_camera=True,
    root_path=None,
    overwrite_root_path=None,
    do_not_check_valid_flag=False,
    load_boxes_3d_in_sweeps=True,
):
    """Generate the train/val/test infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        name_mapping (dict): Mapping from nuScenes category name to target class name.
        test (bool): Whether use the test mode. In the test mode, no annotations can be accessed.
            Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.
        attr_categories_mapper (func): A function that maps attr_categories
            (e.g., `pedestrian_state` -> `pedestrian`).
        include_camera (bool): Whether to include camera information.
        root_path (str): Path of the root data. Default: None.
        overwrite_root_path (str): If not None, replace root_path of lidar_path with this argument.
        do_not_check_valid_flag (bool): Whether to check the valid flag.
        load_boxes_3d_in_sweeps (bool): Whether to load 3D boxes in sweeps.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.

    """
    train_nusc_infos = []
    val_nusc_infos = []
    test_nusc_infos = []

    # for sample in mmcv.track_iter_progress(nusc.sample):
    for sample in mmengine.track_iter_progress(nusc.sample):
        lidar_token = get_lidar_token(sample)
        if lidar_token is None:
            continue
        sd_rec = nusc.get("sample_data", lidar_token)
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        scene_rec = nusc.get("scene", sample["scene_token"])
        log_rec = nusc.get("log", scene_rec["log_token"])

        mmengine.check_file_exist(lidar_path)

        if root_path is not None and overwrite_root_path is not None:
            if lidar_path.startswith("/"):
                lidar_path = lidar_path.replace(root_path, overwrite_root_path)
            else:
                lidar_path = os.path.join(overwrite_root_path, lidar_path)

        info = {
            "scene_name": scene_rec["name"],
            "location": log_rec["location"],
            "lidar_path": lidar_path,
            "scene_token": sample["scene_token"],
            "token": sample["token"],
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        if include_camera:
            # obtain 6 image's information per frame
            camera_types = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            for cam in camera_types:
                cam_token = sample["data"].get(cam)
                # Note(yukke42): some data have missing camera data.
                if cam_token is None:
                    continue

                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                im_sd_rec = nusc.get("sample_data", cam_token)
                cam_infos = []
                while len(cam_infos) <= max_sweeps:
                    # get pose record of the current sample data
                    cam_pose_record = nusc.get("ego_pose", im_sd_rec["ego_pose_token"])
                    cam_info = obtain_sensor2top(
                        nusc,
                        im_sd_rec["token"],
                        l2e_t,
                        l2e_r_mat,
                        cam_pose_record["translation"],
                        Quaternion(cam_pose_record["rotation"]).rotation_matrix,
                        "cam",
                        root_path,
                        overwrite_root_path,
                    )
                    cam_info.update(cam_intrinsic=cam_intrinsic)
                    cam_infos.append(cam_info)
                    if not im_sd_rec["prev"] == "":
                        im_sd_rec = nusc.get("sample_data", im_sd_rec["prev"])
                    else:
                        break
                info["cams"].update({cam: cam_infos})

        # obtain sweeps for a single key-frame
        sweeps = []
        sample_token = sd_rec["sample_token"]
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(
                    nusc,
                    sd_rec["prev"],
                    l2e_t,
                    l2e_r_mat,
                    e2g_t,
                    e2g_r_mat,
                    "lidar",
                    root_path,
                    overwrite_root_path,
                    name_mapping,
                    load_boxes_3d_in_sweeps,
                )
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps
        # obtain annotation
        if not test:
            annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
            instance_tokens = [ann["instance_token"] for ann in annotations]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample["anns"]])
            if not do_not_check_valid_flag:
                valid_flag = np.array(
                    [anno["num_lidar_pts"] > 0 for anno in annotations], dtype=bool
                ).reshape(-1)
            else:
                # NOTE(kan-bayashi): UCv2.0 dataset does not contain meaningful num_lidar_pts,
                #   i.e., all anntations have num_lidar_pts = 0.
                valid_flag = np.array([True for anno in annotations], dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = np.array([name_mapping.get(b.name, b.name) for b in boxes])
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"

            gt_attrs = []
            for anno in annotations:
                if len(anno["attribute_tokens"]) == 0:
                    gt_attrs.append("none")
                else:
                    attr_names = [
                        nusc.get("attribute", t)["name"] for t in anno["attribute_tokens"]
                    ]
                    attr_categories = [a.split(".")[0] for a in attr_names]
                    if attr_categories_mapper("pedestrian_state") in attr_categories:
                        gt_attrs.append(
                            attr_names[
                                attr_categories.index(attr_categories_mapper("pedestrian_state"))
                            ]
                        )
                    elif attr_categories_mapper("cycle_state") in attr_categories:
                        gt_attrs.append(
                            attr_names[
                                attr_categories.index(attr_categories_mapper("cycle_state"))
                            ]
                        )
                    elif attr_categories_mapper("vehicle_state") in attr_categories:
                        gt_attrs.append(
                            attr_names[
                                attr_categories.index(attr_categories_mapper("vehicle_state"))
                            ]
                        )
                    else:
                        raise ValueError(f"invalid attributes: {attr_names}")

            assert len(names) == len(gt_attrs), f"{len(names)}, {len(gt_attrs)}"
            assert len(gt_boxes) == len(instance_tokens)

            info["gt_boxes"] = gt_boxes
            info["instance_tokens"] = instance_tokens
            info["gt_nusc_names"] = [b.name for b in boxes]
            info["gt_names"] = names
            info["gt_attrs"] = np.array(gt_attrs)
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            info["valid_flag"] = valid_flag

            # 2d annotations
            if nuim is not None:
                # key-frame images
                key_im_sd_rec = [
                    o
                    for o in nuim.sample_data
                    if (
                        o["sample_token"] == sample_token
                        and o["fileformat"] != "pcd.bin"
                        and o["is_key_frame"] is True
                    )
                ]  # len=len(camera_type)
                # obtain sweeps for key-frame images
                im_sweeps = []
                while len(im_sweeps) <= len(sweeps):
                    im_sweep = []
                    for im_sd_rec in key_im_sd_rec:
                        if not im_sd_rec["prev"] == "":
                            im_sweep.append(nuim.get("sample_data", im_sd_rec["prev"]))
                        else:
                            continue
                    key_im_sd_rec = im_sweep
                    im_sweeps.append(im_sweep)

                # obtain 2d annotations for sweeps
                camera_label_sweeps = []
                for im_sweep in im_sweeps:
                    camera_annotations = {}
                    camera_annotations.update(
                        {
                            "filename": [],
                            "sample_token": im_sd_rec["sample_token"],
                            "name": [],
                            "bbox": [],
                            "camera_token": [],
                        }
                    )
                    for im_sd_rec in im_sweep:
                        sample_data_token = im_sd_rec["token"]
                        object_anns = [
                            o
                            for o in nuim.object_ann
                            if o["sample_data_token"] == sample_data_token
                        ]
                        if len(object_anns) == 0:
                            continue

                        bbox = np.asarray([o["bbox"][:4] for o in object_anns])
                        drop_list = []
                        for i, box in enumerate(bbox):
                            if box[2] < box[0]:
                                drop_list.append(i)
                                continue
                        camera_type = im_sd_rec["filename"].split("/")[1]
                        camera_token = [camera_type for i in range(len(bbox))]
                        filenames = [
                            str(nusc.get_sample_data_path(im_sd_rec["token"]))
                            for i in range(len(bbox))
                        ]
                        nuim_name = [
                            nuim.get("category", o["category_token"])["name"] for o in object_anns
                        ]
                        names = [name_mapping[c] for c in nuim_name]

                        camera_annotations["filename"].append(np.delete(filenames, drop_list))
                        camera_annotations["name"].append(np.delete(names, drop_list))
                        camera_annotations["bbox"].append(np.delete(bbox, drop_list, 0))
                        camera_annotations["camera_token"].append(
                            np.delete(camera_token, drop_list)
                        )

                    for k, v in camera_annotations.items():
                        if k in ["sample_token"]:
                            continue
                        if k in ["bbox"] and len(v) > 0:
                            camera_annotations[k] = np.vstack(v)
                        elif len(v) > 0:
                            camera_annotations[k] = np.hstack(v)
                    camera_label_sweeps.append(camera_annotations)
                if camera_label_sweeps is not None:
                    info["annos_2d"] = camera_label_sweeps

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        elif sample["scene_token"] in val_scenes:
            val_nusc_infos.append(info)
        elif sample["scene_token"] in test_scenes:
            test_nusc_infos.append(info)
        else:
            # Note: データが増えた場合は一旦学習データとして扱う
            train_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos, test_nusc_infos


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
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
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
