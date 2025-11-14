#!/usr/bin/env python3
import argparse
from collections import defaultdict
import json
import os.path as osp
import pickle

import numpy as np
from nuscenes.eval.common.utils import Box
from pyquaternion import Quaternion
from tqdm import tqdm


def load_pseudo_label(pseudo_label_path):
    print(f"Load pseudo label from {pseudo_label_path}")
    with open(pseudo_label_path, "rb") as f:
        pseudo_label_data = pickle.load(f)
    
    # New format uses 'data_list' instead of 'infos'
    if "data_list" in pseudo_label_data:
        pseudo_label_infos = pseudo_label_data["data_list"]
        metainfo = pseudo_label_data.get("metainfo", {})
    else:
        # Fallback to old format
        pseudo_label_infos = pseudo_label_data.get("infos", [])
        metainfo = pseudo_label_data.get("metainfo", {})
    
    sorted_infos = list(sorted(pseudo_label_infos, key=lambda info: info.get("timestamp", 0)))
    return sorted_infos, metainfo


def box_lidar_2_global(
    lidar_box: Box,
    lidar2ego_rotation: np.ndarray,
    lidar2ego_translation: np.ndarray,
    ego2global_rotation: np.ndarray,
    ego2global_translation: np.ndarray,
) -> Box:
    global_box: Box = lidar_box.copy()
    global_box.rotate(Quaternion(lidar2ego_rotation))
    global_box.translate(np.array(lidar2ego_translation))
    global_box.rotate(Quaternion(ego2global_rotation))
    global_box.translate(np.array(ego2global_translation))
    return global_box


def get_scenes_anno_dict(pseudo_label_infos, metainfo):
    scenes_anno_dict = defaultdict(list)

    # Get label mapping from metainfo
    classes = metainfo.get("classes", ["car", "pedestrian", "bicycle"])
    label_id_to_name = {i: class_name for i, class_name in enumerate(classes)}
    
    instance_ids_dict = {}
    label_ids_count = defaultdict(lambda: 1)

    for pseudo_label_info in tqdm(pseudo_label_infos):
        # Get scene name from various possible keys
        dataset_id = pseudo_label_info.get("scene_name") or pseudo_label_info.get("sample_idx", "unknown")
        
        # Get lidar path
        lidar_path = pseudo_label_info.get("lidar_path", "")
        if lidar_path:
            file_id = f"{lidar_path.split('/')[-1]}.pcd"
        else:
            file_id = f"{dataset_id}.pcd"

        # New format: pred_instances_3d
        pred_instances = pseudo_label_info.get("pred_instances_3d", [])
        
        for pred_instance in pred_instances:
            # Extract data from new format
            bbox_3d = pred_instance["bbox_3d"]  # [x, y, z, x_size, y_size, z_size, yaw]
            velocity = pred_instance["velocity"]  # [vx, vy]
            instance_id = pred_instance["instance_id_3d"]
            bbox_label_id = pred_instance["bbox_label_3d"]
            score = pred_instance["bbox_score_3d"]
            
            # Convert label_id to label_name
            label = label_id_to_name.get(bbox_label_id, "car")

            if instance_id not in instance_ids_dict:
                instance_ids_dict[instance_id] = label_ids_count[label]
                label_ids_count[label] += 1

            # Create Box from bbox_3d
            bbox = Box(
                center=[bbox_3d[0], bbox_3d[1], bbox_3d[2]],
                size=[bbox_3d[3], bbox_3d[4], bbox_3d[5]],
                orientation=Quaternion(axis=[0, 0, 1], radians=-bbox_3d[6] - np.pi / 2),
                velocity=(velocity[0], velocity[1], 0),
                name=label,
                score=score,
                token=instance_id,
            )

            # Transform bbox from calibrated_sensor to global coordinate
            lidar2ego_rotation = Quaternion(pseudo_label_info.get("lidar2ego_rotation", [1, 0, 0, 0]))
            lidar2ego_translation = pseudo_label_info.get("lidar2ego_translation", [0, 0, 0])
            ego2global_rotation = Quaternion(pseudo_label_info.get("ego2global_rotation", [1, 0, 0, 0]))
            ego2global_translation = pseudo_label_info.get("ego2global_translation", [0, 0, 0])
            
            bbox_global = box_lidar_2_global(
                bbox,
                lidar2ego_rotation,
                lidar2ego_translation,
                ego2global_rotation,
                ego2global_translation,
            )
            quaternion = bbox_global.orientation

            scenes_anno_dict[dataset_id].append(
                {
                    "dataset_id": dataset_id,
                    "file_id": file_id,
                    "label_category_id": label,
                    "label_id": f"{label}:{instance_ids_dict[instance_id]}",
                    "instance_id": instance_id,
                    "label_type": "3d_bbox",
                    "attributes": {"pseudo-label": "auto-labeled"},
                    "labeller_email": "pseudo-label@AWML",
                    "sensor_id": "lidar",
                    "three_d_bbox": {
                        "cx": float(bbox_global.center[0]),
                        "cy": float(bbox_global.center[1]),
                        "cz": float(bbox_global.center[2]),
                        "h": float(bbox_global.wlh[2]),
                        "l": float(bbox_global.wlh[1]),
                        "w": float(bbox_global.wlh[0]),
                        "quaternion": {
                            "x": float(quaternion.x),
                            "y": float(quaternion.y),
                            "z": float(quaternion.z),
                            "w": float(quaternion.w),
                        },
                    },
                }
            )
    return scenes_anno_dict


def save_deepen_json(scenes_anno_dict, output_dir):
    """Save annotations in Deepen format."""
    for scene_name, scene_anno_dict in scenes_anno_dict.items():
        file_name = osp.join(output_dir, f"{scene_name}.json")
        print(f"Generate deepen format json: {file_name}")
        deepen_anno_json = {"labels": scene_anno_dict}
        with open(file_name, "w") as f:
            json.dump(deepen_anno_json, f, indent=4)


def save_segment_ai_json(scenes_anno_dict, output_dir):
    """Save annotations in Segment.ai format (placeholder for future implementation)."""
    raise NotImplementedError("Segment.ai format is not yet supported")


def convert_pseudo_to_annotation_format(pseudo_label_path, output_dir, output_format):
    """Convert pseudo label to specified annotation tool format.
    
    Args:
        pseudo_label_path: Path to pseudo label pickle file
        output_dir: Output directory for annotation files
        output_format: Output format ('deepen' or 'segment.ai')
    """
    pseudo_label_infos, metainfo = load_pseudo_label(pseudo_label_path)
    scenes_anno_dict = get_scenes_anno_dict(pseudo_label_infos, metainfo)
    
    if output_format == "deepen":
        save_deepen_json(scenes_anno_dict, output_dir)
    elif output_format == "segment.ai":
        save_segment_ai_json(scenes_anno_dict, output_dir)
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Supported formats: 'deepen', 'segment.ai'")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert pseudo label to annotation tool format (e.g., Deepen, Segment.ai)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to Pseudo Label pickle file (info.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        required=True,
        choices=["deepen", "segment.ai"],
        help="Output format for annotation tool (currently only 'deepen' is supported)",
    )
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    convert_pseudo_to_annotation_format(
        args.input, 
        args.output_dir, 
        args.output_format
    )


if __name__ == "__main__":
    main()
