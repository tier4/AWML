# Copyright (c) TIER IV, inc. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import base64
import json
import os
import re
import warnings
from collections import OrderedDict
from functools import partial
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import mmengine
import numpy as np
import yaml
from nuimages import NuImages
from nuimages.utils.utils import mask_decode, name_to_index_mapping
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

autoware_categories = (
    "UNKNOWN",
    "CAR",
    "TRUCK",
    "BUS",
    "BICYCLE",
    "MOTORBIKE",
    "PEDESTRIAN",
    "ANIMAL",
)

NAME_MAPPING = {
    "vehicle.car": "CAR",
    "human.pedestrian.adult": "PEDESTRIAN",
    "movable_object.barrier": "UNKNOWN",
    "movable_object.traffic_cone": "UNKNOWN",
    "vehicle.truck": "TRUCK",
    "vehicle.bicycle": "BICYCLE",
    "vehicle.motorcycle": "MOTORBIKE",
    "human.pedestrian.construction_worker": "PEDESTRIAN",
    "vehicle.bus.rigid": "BUS",
    "vehicle.bus.bendy": "BUS",
    "vehicle.construction": "CAR",
    "vehicle.trailer": "CAR",
    "movable_object.pushable_pullable": "UNKNOWN",
    "movable_object.debris": "UNKNOWN",
    "static_object.bicycle rack": "BICYCLE",
    "human.pedestrian.personal_mobility": "PEDESTRIAN",
    "human.pedestrian.child": "PEDESTRIAN",
    "human.pedestrian.police_officer": "PEDESTRIAN",
    "human.pedestrian.stroller": "PEDESTRIAN",
    "animal": "ANIMAL",
    "vehicle.emergency.police": "CAR",
    "vehicle.emergency.ambulance": "CAR",
    "human.pedestrian.wheelchair": "PEDESTRIAN",
    "vehicle.ego": "UNKNOWN",
    "static_object.bollard": "UNKNOWN",
    # tier4 data original
    "pedestrian.adult": "PEDESTRIAN",
    "pedestrian.construction_worker": "PEDESTRIAN",
    "vehicle.bus": "BUS",
    "pedestrian.child": "PEDESTRIAN",
    "pedestrian.police_officer": "PEDESTRIAN",
    "pedestrian.stroller": "PEDESTRIAN",
    "vehicle.police": "CAR",
    "vehicle.police_car": "CAR",
    "vehicle.ambulance": "CAR",
    "vehicle.fire": "CAR",
    "pedestrian.wheelchair": "PEDESTRIAN",
    # DBV1-0
    "bus": "BUS",
    "car": "CAR",
    "motorcycle": "MOTORBIKE",
    "pedestrian": "PEDESTRIAN",
    "truck": "TRUCK",
    "bicycle": "BICYCLE",
    "trailer": "TRUCK",
    "police_car": "CAR",
    # DBv2.0 and DBv3.0
    "pedestrian.personal_mobility": "PEDESTRIAN",
    "pedestrian.child": "PEDESTRIAN",
    "pedestrian.police_officer": "PEDESTRIAN",
    "pedestrian.stroller": "PEDESTRIAN",
}


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
                (
                    "The directory structure of T4 Dataset is deprecated."
                    "In the newer version, the directory"
                    " structure should look something like `$T4DATASET_ID/$VERSION_ID/`."
                    " Please update your Web.Auto CLI to the latest version."
                ),
                DeprecationWarning,
            )
            nusc_root_path = root_path
    else:
        nusc_root_path = root_path
    return nusc_root_path


def parse_args():
    parser = argparse.ArgumentParser(description="Create coco annotation file")

    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--version",
        type=str,
        nargs="+",
        default=["v1.0-mini"],
        required=False,
        help="specify the dataset version",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        required=False,
        help="specify the dataset version",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train.json",
        required=False,
        help="specify the split file",
    )
    parser.add_argument(
        "--n-proc",
        type=int,
        default=1,
        required=False,
        help="number of process",
    )
    parser.add_argument(
        "--create-empty-object-ann",
        action="store_true",
        help="create an empty object_ann.json file for generating pseudo-labels",
    )
    parser.add_argument(
        "--occlusion-list",
        type=str,
        nargs="+",
        default=[],
        required=False,
        help="Only the states included in the list will be extracted.",
    )

    args = parser.parse_args()

    return args


def export_t4_to_coco(data_root, version, mode, split, nproc, create_empty_object_ann: bool, occlusion_states=[]):
    """Export 2d annotation from the info file and raw data.
    Args:
        data_root (str): Root path of the raw data.
        version (str): Dataset version.
    """
    with open(split) as f:
        splits = yaml.safe_load(f)[mode]

    categories = [dict(id=autoware_categories.index(cat_name), name=cat_name) for cat_name in autoware_categories]
    cat2id = {k_v["name"]: k_v["id"] for k_v in categories}

    img_idx = 0
    images = []
    # Determine the index of object annotation
    print("Process annotation information...")
    annotations = []
    max_cls_ids = []
    seg_root = None
    for s in splits:
        print(s)
        split_root = search_version_if_exists(os.path.join(data_root, s))
        assert osp.exists(split_root), f"T4Dataset Not Found: {split_root}"
        obj_ann_file = osp.join(split_root, "annotation", "object_ann.json")
        if create_empty_object_ann and not osp.exists(obj_ann_file):
            mmcv.dump([], obj_ann_file)
        nuim = NuImages(version="annotation", dataroot=split_root, verbose=True)
        start_idx = len(images)
        outputs = []
        for sample_info in mmengine.utils.progressbar.track_iter_progress(nuim.sample_data):
            if sample_info["is_key_frame"] and sample_info["fileformat"] != "pcd.bin":
                img_idx = len(images)
                img_info = dict(
                    id=img_idx,
                    token=sample_info["token"],
                    file_name=os.path.join(os.path.relpath(split_root, data_root), sample_info["filename"]),
                    width=sample_info["width"],
                    height=sample_info["height"],
                )
                images.append(img_info)
                if nproc <= 1:
                    outputs.append(
                        get_img_annos(
                            img_info=img_info,
                            nuim=nuim,
                            cat2id=cat2id,
                            seg_root=seg_root,
                            occlusion_states=occlusion_states,
                        )
                    )
        end_idx = len(images)

        print("Process img annotations...")
        if nproc > 1:
            outputs = mmcv.track_parallel_progress(
                partial(get_img_annos, nuim=nuim, cat2id=cat2id, seg_root=seg_root),
                images[start_idx:end_idx],
                nproc=nproc,
            )

        for single_img_annos, max_cls_id in outputs:
            max_cls_ids.append(max_cls_id)
            for img_anno in single_img_annos:
                img_anno.update(id=len(annotations))
                annotations.append(img_anno)

    max_cls_id = max(max_cls_ids)
    print(f"Max ID of class in the semantic map: {max_cls_id}")

    # cleanup unused images
    img_id_map = {}
    new_id = 1
    for ann in annotations:
        if ann["image_id"] not in img_id_map:
            img_id_map[ann["image_id"]] = new_id
            ann["image_id"] = new_id
            new_id += 1
        else:
            ann["image_id"] = img_id_map[ann["image_id"]]

    new_images = []
    for img in images:
        if img["id"] in img_id_map:
            img["id"] = img_id_map[img["id"]]
            new_images.append(img)

    coco_format_json = dict(images=new_images, annotations=annotations, categories=categories)

    mmengine.dump(
        coco_format_json,
        os.path.join(
            data_root,
            version,
            "_".join(occlusion_states),
            "coco_annotations_" + os.path.basename(split).replace(".yaml", ".json"),
        ),
    )


def get_2dobj_boxes(nusc, sample_data_token: str, cat2id):
    """Get the 2D annotation records for a given `sample_data_token`.
    Args:
        sample_data_token (str): Sample data token belonging to a camera
            keyframe.
        visibilities (list[str]): Visibility filter.
    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Load object instances.
    object_anns = [o for o in nusc.object_ann if o["sample_data_token"] == sample_data_token]

    # Sort by token to ensure that objects always appear in the
    # instance mask in the same order.
    object_anns = sorted(object_anns, key=lambda k: k["token"])

    # Draw object instances.
    # The 0 index is reserved for background; thus, the instances
    # should start from index 1.
    annotations = []
    for i, ann in enumerate(object_anns, start=1):
        # Get box
        # The segmentation class number(24) is hard-coded in cocodataset.
        # T4 has more, thus segmentation anno could not get loaded by now.
        category_token = ann["category_token"]
        category_name = nusc.get("category", category_token)["name"]

        if category_name in NAME_MAPPING:
            cat_name = NAME_MAPPING[category_name]
            cat_id = cat2id[cat_name]

            x_min, y_min, x_max, y_max, *_ = ann["bbox"]

            data_anno = dict(
                image_id=sample_data_token,
                category_id=cat_id,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                iscrowd=0,
            )
            annotations.append(data_anno)
    return annotations


def get_2d_boxes(nusc, sample_data_token: str, visibilities: List[str]):
    """Get the 2D annotation records for a given `sample_data_token`.
    Args:
        sample_data_token (str): Sample data token belonging to a camera
            keyframe.
        visibilities (list[str]): Visibility filter.
    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get("sample_data", sample_data_token)

    assert sd_rec["sensor_modality"] == "camera", "Error: get_2d_boxes only works" " for camera sample_data!"
    if not sd_rec["is_key_frame"]:
        raise ValueError("The 2D re-projections are available only for keyframes.")

    s_rec = nusc.get("sample", sd_rec["sample_token"])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    camera_intrinsic = np.array(cs_rec["camera_intrinsic"])

    # Get all the annotation with the specified visibilties.
    ann_recs = [nusc.get("sample_annotation", token) for token in s_rec["anns"]]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec["visibility_token"] in visibilities)]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec["sample_annotation_token"] = ann_rec["token"]
        ann_rec["sample_data_token"] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec["token"])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec["translation"]))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec["translation"]))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec["filename"])

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.
    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.
    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(
    ann_rec: dict,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    sample_data_token: str,
    filename: str,
) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.
    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.
    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec["sample_data_token"] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        "attribute_tokens",
        "category_name",
        "instance_token",
        "next",
        "num_lidar_pts",
        "num_radar_pts",
        "prev",
        "sample_annotation_token",
        "sample_data_token",
        "visibility_token",
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec["bbox_corners"] = [x1, y1, x2, y2]
    repro_rec["filename"] = filename

    coco_rec["file_name"] = filename
    coco_rec["image_id"] = sample_data_token
    coco_rec["area"] = (y2 - y1) * (x2 - x1)

    if repro_rec["category_name"] not in NAME_MAPPING:
        return None
    cat_name = NAME_MAPPING[repro_rec["category_name"]]
    coco_rec["category_name"] = cat_name
    coco_rec["category_id"] = autoware_categories.index(cat_name)
    coco_rec["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec["iscrowd"] = 0

    return coco_rec


def get_img_annos(img_info, nuim, cat2id, seg_root, occlusion_states):
    """Get semantic segmentation map for an image.
    Args:
        nuim (obj:`NuImages`): NuImages dataset object
        img_info (dict): Meta information of img
    Returns:
        np.ndarray: Semantic segmentation map of the image
    """
    # print (img_info)
    sd_token = img_info["token"]
    image_id = img_info["id"]

    # Get image data.
    width, height = img_info["width"], img_info["height"]
    semseg_mask = np.zeros((height, width)).astype("uint8")

    if seg_root is not None:
        # Load stuff / surface regions.
        surface_anns = [o for o in nuim.surface_ann if o["sample_data_token"] == sd_token]

        # Draw stuff / surface regions.
        for ann in surface_anns:
            # Get color and mask.
            category_token = ann["category_token"]
            category_name = nuim.get("category", category_token)["name"]
            if ann["mask"] is None:
                continue
            mask = mask_decode(ann["mask"])

            # Draw mask for semantic segmentation.
            name_to_index = name_to_index_mapping(nuim.category)
            semseg_mask[mask == 1] = name_to_index[category_name]

    # Load object instances.
    object_anns = [o for o in nuim.object_ann if o["sample_data_token"] == sd_token]

    # Sort by token to ensure that objects always appear in the
    # instance mask in the same order.
    object_anns = sorted(object_anns, key=lambda k: k["token"])

    # Draw object instances.
    # The 0 index is reserved for background; thus, the instances
    # should start from index 1.
    annotations = []
    for i, ann in enumerate(object_anns, start=1):
        # Get color, box, mask and name.
        category_token = ann["category_token"]
        category_name = nuim.get("category", category_token)["name"]

        # Get attributes
        attribute_tokens = ann["attribute_tokens"]

        attribute_names = []
        for token in attribute_tokens:
            attribute_names.append(nuim.get("attribute", token)["name"])

        # Filter by occlusion states
        # Do not skip if no attributes are present, such as UNKNOWN.
        if len(occlusion_states) > 0 and len(attribute_names) > 0:
            check_status = False
            for name in attribute_names:
                if name in occlusion_states:
                    check_status = True
            if not check_status:
                continue

        if ann["mask"] is None:
            continue
        mask = mask_decode(ann["mask"])

        # Draw masks for semantic segmentation and instance segmentation.
        if seg_root is not None:
            semseg_mask[mask == 1] = name_to_index[category_name]

        if category_name in NAME_MAPPING:
            cat_name = NAME_MAPPING[category_name]
            cat_id = cat2id[cat_name]

            x_min, y_min, x_max, y_max, *_ = ann["bbox"]
            # encode calibrated instance mask
            mask_anno = dict()
            mask_anno["counts"] = base64.b64decode(ann["mask"]["counts"]).decode()
            mask_anno["size"] = ann["mask"]["size"]

            data_anno = dict(
                image_id=image_id,
                category_id=cat_id,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=mask_anno,
                attribute=attribute_names,
                iscrowd=0,
            )
            annotations.append(data_anno)

    # after process, save semantic masks
    img_filename = img_info["file_name"]
    if seg_root is not None:
        seg_filename = img_filename.replace("jpg", "png")
        seg_filename = osp.join(seg_root, seg_filename)
        mmcv.imwrite(semseg_mask, seg_filename)
    return annotations, np.max(semseg_mask)


def main():
    args = parse_args()

    occlusion_states = [f"occlusion_state.{name}" for name in args.occlusion_list]

    for version in args.version:
        export_t4_to_coco(
            args.root_path,
            version,
            args.mode,
            args.splits,
            args.n_proc,
            args.create_empty_object_ann,
            occlusion_states,
        )


if __name__ == "__main__":
    main()
