import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import mmengine
import numpy as np
import numpy.typing as npt
from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from nuimages import NuImages
from nuscenes import NuScenes
from nuscenes.nuscenes import Box
from pyquaternion import Quaternion
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import translate as shapely_translate
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

from tools.detection3d.t4dataset_converters.update_infos_to_v2 import (
    clear_instance_unused_keys,
    get_empty_instance,
    get_single_image_sweep,
    get_single_lidar_sweep,
)


def get_ego2global(pose_record: Dict) -> Dict[str, List]:
    ego2global = convert_quaternion_to_matrix(
        quaternion=pose_record["rotation"], translation=pose_record["translation"]
    )
    return dict(
        ego2global=ego2global,
    )


def parse_camera_path(camera_path: str) -> str:
    """leave only {database_version}/{scene_id}/{dataset_version}/data/{camera_type}/{frame}.bin from path"""
    return "/".join(camera_path.split("/")[-6:])


def parse_lidar_path(lidar_path: str) -> str:
    """leave only {database_version}/{scene_id}/{dataset_version}/data/{lidar_token}/{frame}.bin from path"""
    return "/".join(lidar_path.split("/")[-6:])


def get_lidar_points_info(
    lidar_path: str,
    cs_record: Dict,
    num_features: int = 5,
):
    lidar2ego = convert_quaternion_to_matrix(quaternion=cs_record["rotation"], translation=cs_record["translation"])
    mmengine.check_file_exist(lidar_path)
    lidar_path = parse_lidar_path(lidar_path)
    return dict(
        lidar_points=dict(
            num_pts_feats=num_features,
            lidar_path=lidar_path,
            lidar2ego=lidar2ego,
        ),
    )


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
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
    mmengine.check_file_exist(lidar_path)

    scene_record = nusc.get("scene", sample["scene_token"])
    log_record = nusc.get("log", scene_record["log_token"])

    l2e_t = cs_record["translation"]
    e2g_t = pose_record["translation"]
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
        e2g_t,
        l2e_t,
    )


def get_gt_attrs(nusc, annotations) -> List[List[str]]:
    gt_attrs = []
    for anno in annotations:
        gt_attrs.append([nusc.get("attribute", t)["name"] for t in anno["attribute_tokens"]])
    return gt_attrs


def check_boxes_overlap(box1, box2):
    """
    Check if two 3D bounding boxes overlap in 2D projection.
    Args:
        box1 (np.ndarray): Bounding box 1 in the format (x, y, z, dx, dy, dz, yaw).
        box2 (np.ndarray): Bounding box 2 in the format (x, y, z, dx, dy, dz, yaw).
    """

    def get_corners(box):
        x, y, z, dx, dy, dz, yaw = box
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        half_dx = dx / 2
        half_dy = dy / 2

        corners = np.array(
            [
                [x - half_dx * cos_yaw + half_dy * sin_yaw, y - half_dx * sin_yaw - half_dy * cos_yaw],
                [x + half_dx * cos_yaw + half_dy * sin_yaw, y + half_dx * sin_yaw - half_dy * cos_yaw],
                [x + half_dx * cos_yaw - half_dy * sin_yaw, y + half_dx * sin_yaw + half_dy * cos_yaw],
                [x - half_dx * cos_yaw - half_dy * sin_yaw, y - half_dx * sin_yaw + half_dy * cos_yaw],
            ],
        )

        return corners

    corners1 = get_corners(box1)
    corners2 = get_corners(box2)

    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)

    return poly1.intersects(poly2)


def check_boxes_proximity(box1, box2, distance_threshold=1.0):

    def get_face_centers(box):
        x, y, z, dx, dy, dz, yaw = box
        front_center = np.array([x + dx / 2 * np.cos(yaw), y + dx / 2 * np.sin(yaw), z])
        back_center = np.array([x - dx / 2 * np.cos(yaw), y - dx / 2 * np.sin(yaw), z])
        return front_center, back_center

    front1, back1 = get_face_centers(box1)
    front2, back2 = get_face_centers(box2)

    if np.linalg.norm(front1 - front2) <= distance_threshold or np.linalg.norm(front1 - back2) <= distance_threshold:
        return True
    if np.linalg.norm(back1 - front2) <= distance_threshold or np.linalg.norm(back1 - back2) <= distance_threshold:
        return True

    return False


def match_objects(
    gt_bboxes_3d: np.ndarray,
    gt_names_3d: np.ndarray,
    merge_objects: List[Tuple[str, List[str]]],
) -> List[tuple]:
    """
    Match objects according to merge_objects criteria.
    Args:
        gt_bboxes_3d (np.ndarray): Ground truth bounding boxes.
        gt_names_3d (np.ndarray[str]): Ground truth names.
        merge_objects (List[Tuple[str, List[str]]]): List of tuples specifying objects to merge.
    Returns:
        matched_object_idxs (List[tuple]): List of matched object indices.
    """
    matched_pairs = {i: [] for i, _ in merge_objects}
    distance_threshold = 2  # Adjust this threshold as needed. in meters

    for target_object, sub_objects in merge_objects:
        pairs = []
        sub_object1_indices = np.where(np.isin(gt_names_3d, sub_objects[0]))[0]
        sub_object2_indices = np.where(np.isin(gt_names_3d, sub_objects[1]))[0]

        sub_object1_bboxes = gt_bboxes_3d[sub_object1_indices]
        sub_object2_bboxes = gt_bboxes_3d[sub_object2_indices]

        for idx1, box1 in zip(sub_object1_indices, sub_object1_bboxes):
            for idx2, box2 in zip(sub_object2_indices, sub_object2_bboxes):
                if check_boxes_overlap(box1, box2) or check_boxes_proximity(box1, box2, distance_threshold):
                    pairs.append((idx1, idx2))
        # if len(pairs) != len(sub_object2_indices):
        #     print(f"WARNING: Did not find matching pairs for all {sub_objects}. {sub_objects[0]}: {len(sub_object1_indices)},{sub_objects[1]}: {len(sub_object2_indices)}, found {len(pairs)} pairs")
        matched_pairs[target_object].extend(pairs)
    return matched_pairs


def merge_boxes_union(box1: List[float], box2: List[float]):
    """
    Merges two 3D bounding boxes by calculating the union of their projections on the XY plane.

    The function creates 2D representations of the input 3D boxes using the Shapely library,
    then computes their union, represented as the minimum rotated rectangle that contains both boxes.
    The resulting 3D bounding box has the combined dimensions and orientation of the merged box.
    Compared to the `merge_boxes_extend_longer method`, this merging method tends to shift the
    centroid of trailers+tractors merges more during turns.

    Parameters:
        box1 (List[float]): A list representing the first 3D box with the format
                            [x, y, z, dx, dy, dz, yaw], where:
                            - x, y, z: Center coordinates of the box.
                            - dx, dy, dz: Dimensions of the box along the x, y, and z axes.
                            - yaw: Rotation of the box around the z-axis in radians.
        box2 (List[float]): A list representing the second 3D box with the same format as box1.

    Returns:
        List[float]: A list representing the merged 3D box with the same format as bbox1,2
    """

    print("Box merging strategy: union (from merge_boxes_union)")

    def get_shapely_box(box):
        x, y, z, dx, dy, dz, yaw = box
        # Create a 2D shapely box
        shapely_box_object = shapely_box(-dx / 2, -dy / 2, dx / 2, dy / 2)
        # Rotate and translate the box
        shapely_box_object = shapely_rotate(shapely_box_object, yaw, origin=(0, 0), use_radians=True)
        shapely_box_object = shapely_translate(shapely_box_object, x, y)
        return shapely_box_object, z, dz, yaw

    box1_shapely, z1, dz1, yaw1 = get_shapely_box(box1)
    box2_shapely, z2, dz2, yaw2 = get_shapely_box(box2)

    # Merge the two shapely boxes
    merged_box = unary_union([box1_shapely, box2_shapely]).minimum_rotated_rectangle

    # Get the coordinates of the rectangle
    coords = list(merged_box.exterior.coords)[:-1]  # Exclude the repeated first/last point

    # Calculate the new dimensions and center
    new_x = sum([point[0] for point in coords]) / 4
    new_y = sum([point[1] for point in coords]) / 4

    edge1 = ((coords[0][0] - coords[1][0]) ** 2 + (coords[0][1] - coords[1][1]) ** 2) ** 0.5
    edge2 = ((coords[1][0] - coords[2][0]) ** 2 + (coords[1][1] - coords[2][1]) ** 2) ** 0.5

    new_dx = max(edge1, edge2)
    new_dy = min(edge1, edge2)

    new_z = min(z1, z2)
    new_dz = max(z1 + dz1, z2 + dz2) - new_z

    # Calculate the new orientation
    if edge1 >= edge2:
        new_yaw = np.arctan2(
            coords[1][1] - coords[0][1],
            coords[1][0] - coords[0][0],
        )
    else:
        new_yaw = np.arctan2(
            coords[2][1] - coords[1][1],
            coords[2][0] - coords[1][0],
        )

    return [new_x, new_y, new_z, new_dx, new_dy, new_dz, new_yaw]


def merge_boxes_extend_longer(box1: List[float], box2: List[float]):
    """
    Gives impression of merging two 3D bounding boxes by elongating the larger box.

    The function identifies the larger and smaller box based on their area in the XY plane.
    The center of the farther end of the smaller box is rotated to meet the length axis of the
    larger box. Then, the larger box is elongated upto that point.
    https://docs.google.com/presentation/d/17802H6gqApU3mHN2Q5XUcqa_qR5y5a_76QMM2F_9WW8/edit#slide=id.g20a727e0846_3_0

    Parameters:
        box1 (List[float]): A list representing the first 3D box with the format
                            [x, y, z, dx, dy, dz, yaw], where:
                            - x, y, z: Center coordinates of the box.
                            - dx, dy, dz: Dimensions of the box along the x, y, and z axes.
                            - yaw: Rotation of the box around the z-axis in radians.
        box2 (List[float]): A list representing the second 3D box with the same format as box1.

    Returns:
        List[float]: A list representing the merged 3D box with the same format as bbox1,2
    """

    print("Box merging strategy: longer (from merge_boxes_extend_longer)")

    def get_box_faces(box):
        x, y, z, dx, dy, dz, yaw = box
        center = np.array([x, y])
        if dx >= dy:
            face1_center = np.array([x + (dx / 2) * np.cos(yaw), y + (dx / 2) * np.sin(yaw)])
            face2_center = np.array([x - (dx / 2) * np.cos(yaw), y - (dx / 2) * np.sin(yaw)])
        else:
            face1_center = np.array([x + (dy / 2) * np.cos(yaw + np.pi / 2), y + (dy / 2) * np.sin(yaw + np.pi / 2)])
            face2_center = np.array([x - (dy / 2) * np.cos(yaw + np.pi / 2), y - (dy / 2) * np.sin(yaw + np.pi / 2)])
        return center, face1_center, face2_center, dx, dy

    # Identify the centers and faces of both boxes
    box1_center, box1_face1, box1_face2, dx1, dy1 = get_box_faces(box1)
    box2_center, box2_face1, box2_face2, dx2, dy2 = get_box_faces(box2)

    # Determine which box is larger
    if dx1 * dy1 >= dx2 * dy2:
        larger_box_center, larger_box_face1, larger_box_face2, larger_dx, larger_dy, larger_box = (
            box1_center,
            box1_face1,
            box1_face2,
            dx1,
            dy1,
            box1,
        )
        smaller_box_center, smaller_box_face1, smaller_box_face2, smaller_dx, smaller_dy = (
            box2_center,
            box2_face1,
            box2_face2,
            dx2,
            dy2,
        )
    else:
        larger_box_center, larger_box_face1, larger_box_face2, larger_dx, larger_dy, larger_box = (
            box2_center,
            box2_face1,
            box2_face2,
            dx2,
            dy2,
            box2,
        )
        smaller_box_center, smaller_box_face1, smaller_box_face2, smaller_dx, smaller_dy = (
            box1_center,
            box1_face1,
            box1_face2,
            dx1,
            dy1,
        )

    # Choose the farther face of the smaller box
    dist_to_smaller_face1 = np.linalg.norm(smaller_box_face1 - larger_box_center)
    dist_to_smaller_face2 = np.linalg.norm(smaller_box_face2 - larger_box_center)
    if dist_to_smaller_face1 > dist_to_smaller_face2:
        selected_smaller_face = smaller_box_face1
    else:
        selected_smaller_face = smaller_box_face2

    # Choose the nearer face of the larger box
    dist_to_larger_face1 = np.linalg.norm(larger_box_face1 - smaller_box_center)
    dist_to_larger_face2 = np.linalg.norm(larger_box_face2 - smaller_box_center)
    if dist_to_larger_face1 < dist_to_larger_face2:
        selected_larger_face = larger_box_face1
    else:
        selected_larger_face = larger_box_face2

    # Find the projection point on the axis of the larger box
    axis_vector = selected_larger_face - larger_box_center
    axis_vector_normalized = axis_vector / np.linalg.norm(axis_vector)
    to_smaller_box = selected_smaller_face - larger_box_center
    projection_length = np.dot(to_smaller_box, axis_vector_normalized)
    projection_point = larger_box_center + projection_length * axis_vector_normalized

    # Elongate the larger box to the projection point
    elongation_vector = projection_point - selected_larger_face
    elongation_length = np.linalg.norm(elongation_vector)

    new_dx = larger_dx + elongation_length if larger_dx >= larger_dy else larger_dx
    new_dy = larger_dy + elongation_length if larger_dy > larger_dx else larger_dy

    # Adjust the center minimally to balance the elongation
    elongation_shift = elongation_vector / 2
    new_center = larger_box_center + elongation_shift

    new_z = min(box1[2], box2[2])
    new_dz = max(box1[2] + box1[5], box2[2] + box2[5]) - new_z

    # Keep the orientation (yaw) of the larger box
    new_yaw = larger_box[6]

    return [
        new_center[0],
        new_center[1],
        new_z,
        new_dx,
        new_dy,
        new_dz,
        new_yaw,
    ]


def get_instances(
    gt_boxes,
    names,
    class_names,
    velocity,
    boxes,
    annotations,
    valid_flag,
    gt_attrs,
    filter_attributions: Optional[List[Tuple[str, str]]],
    matched_object_idx=None,
    merge_type="extend_longer",
):

    if merge_type == "extend_longer":
        merge_function = merge_boxes_extend_longer
    elif merge_type == "union":
        merge_function = merge_boxes_union
    else:
        # matching will be skipped in this case
        matched_object_idx = None

    instances = []
    ignore_class_name = set()
    merged_indices = set()
    if matched_object_idx is not None:
        for target_object, pairs in matched_object_idx.items():
            for idx1, idx2 in pairs:
                if idx1 in merged_indices or idx2 in merged_indices:
                    continue
                # Merge the bounding boxes
                new_bbox_3d = merge_function(gt_boxes[idx1], gt_boxes[idx2])
                new_velocity = (velocity[idx1] + velocity[idx2]) / 2
                new_num_lidar_pts = annotations[idx1]["num_lidar_pts"] + annotations[idx2]["num_lidar_pts"]
                new_num_radar_pts = annotations[idx1]["num_radar_pts"] + annotations[idx2]["num_radar_pts"]
                new_attrs = list(set(gt_attrs[idx1] + gt_attrs[idx2]))
                empty_instance = get_empty_instance()

                if target_object in class_names:
                    is_filter = False
                    if filter_attributions:
                        for filter_attribution in filter_attributions:
                            if target_object == filter_attribution[0] and filter_attribution[1] in new_attrs:
                                is_filter = True
                    if is_filter is True:
                        empty_instance["bbox_label"] = -1
                    else:
                        empty_instance["bbox_label"] = class_names.index(target_object)

                else:
                    ignore_class_name.add(target_object)
                    empty_instance["bbox_label"] = -1
                empty_instance["bbox_label_3d"] = copy.deepcopy(empty_instance["bbox_label"])
                empty_instance["bbox_3d"] = new_bbox_3d
                empty_instance["velocity"] = new_velocity.tolist()
                empty_instance["num_lidar_pts"] = new_num_lidar_pts
                empty_instance["num_radar_pts"] = new_num_radar_pts
                empty_instance["bbox_3d_isvalid"] = valid_flag[idx1] and valid_flag[idx2]
                empty_instance["gt_nusc_name"] = target_object
                empty_instance["gt_attrs"] = new_attrs
                # empty_instance["merged_from"] = [gt_boxes[idx1], gt_boxes[idx2]]   # used for debugging, keep commented out otherwise
                empty_instance = clear_instance_unused_keys(empty_instance)

                instances.append(empty_instance)
                merged_indices.add(idx1)
                merged_indices.add(idx2)

    for i, box in enumerate(gt_boxes):
        if i in merged_indices:
            continue
        empty_instance = get_empty_instance()
        empty_instance["bbox_3d"] = box.tolist()
        if names[i] in class_names:
            is_filter = False
            if filter_attributions:
                for filter_attribution in filter_attributions:
                    # If the ground truth name matches exatcly the filtered label name, and
                    # the filtered attribute is in one of the available attribute names
                    if boxes[i].name == filter_attribution[0] and filter_attribution[1] in gt_attrs[i]:
                        is_filter = True
            if is_filter is True:
                empty_instance["bbox_label"] = -1
            else:
                empty_instance["bbox_label"] = class_names.index(names[i])

        else:
            ignore_class_name.add(names[i])
            empty_instance["bbox_label"] = -1
        empty_instance["bbox_label_3d"] = copy.deepcopy(empty_instance["bbox_label"])

        empty_instance["velocity"] = velocity[i].tolist()
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
    filter_attributes: Optional[List[Tuple[str, str]]],
    merge_objects: List[Tuple[str, List[str]]] = [],
    merge_type: str = None,
) -> dict:
    annotations = [nusc.get("sample_annotation", token) for token in anns]
    instance_tokens = [ann["instance_token"] for ann in annotations]
    locs = np.array([b.center for b in boxes]).reshape(-1, 3)
    dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
    rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
    velocity = np.array([nusc.box_velocity(token)[:2] for token in anns])

    valid_flag = np.array([anno["num_lidar_pts"] > 0 for anno in annotations], dtype=bool).reshape(-1)

    # convert velo from global to lidar
    for i in range(len(boxes)):
        velo = np.array([*velocity[i], 0.0])
        velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        velocity[i] = velo[:2]

    names = np.array([name_mapping.get(b.name, b.name) for b in boxes])
    # we need to convert rot to SECOND format.
    # Copied from https://github.com/open-mmlab/mmdetection3d/blob/0f9dfa97a35ef87e16b700742d3c358d0ad15452/tools/dataset_converters/nuscenes_converter.py#L258
    gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
    assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"

    gt_attrs = get_gt_attrs(nusc, annotations)
    assert len(names) == len(gt_attrs), f"{len(names)}, {len(gt_attrs)}"
    assert len(gt_boxes) == len(instance_tokens)
    assert velocity.shape == (len(gt_boxes), 2)

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
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
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
        velocity = np.array([nusc.box_velocity(ann_token)[:2] for ann_token in ann_tokens]).reshape(-1, 2)
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
