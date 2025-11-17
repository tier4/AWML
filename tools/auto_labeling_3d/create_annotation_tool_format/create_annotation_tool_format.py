#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path
import pickle
from collections import defaultdict

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from t4_devkit.dataclass import Box3D as T4Box3D, SemanticLabel, Shape, ShapeType


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
    lidar_box: T4Box3D,
    ego2global: np.ndarray,
) -> T4Box3D:
    # ego2global: 4x4 matrix
    global_box: T4Box3D = copy.deepcopy(lidar_box)
    rot = ego2global[:3, :3]
    trans = ego2global[:3, 3]
    # Use rtol/atol to allow for small non-orthogonality
    global_box.rotate(Quaternion(matrix=rot, rtol=1e-5, atol=1e-7))
    global_box.translate(trans)
    return global_box


def get_scenes_anno_dict(pseudo_label_infos, metainfo, dataset_id=None):
    scenes_anno_dict = defaultdict(list)

    # Get label mapping from metainfo
    classes = metainfo.get("classes", ["car", "pedestrian", "bicycle"])
    label_id_to_name = {i: class_name for i, class_name in enumerate(classes)}
    
    instance_ids_dict = {}
    label_ids_count = defaultdict(lambda: 1)

    for idx, pseudo_label_info in enumerate(tqdm(pseudo_label_infos)):
        # Use provided dataset_id if given, else fallback to scene_name/sample_idx
        scene_id = dataset_id if dataset_id is not None else (
            pseudo_label_info.get("scene_name") or pseudo_label_info.get("sample_idx", "unknown")
        )
        # file_id as sequential number: 0.pcd, 1.pcd, ...
        file_id = f"{idx}.pcd"

        # New format: pred_instances_3d
        pred_instances = pseudo_label_info.get("pred_instances_3d", [])
        
        for pred_instance in pred_instances:
            # Extract data from new format
            bbox_3d = pred_instance["bbox_3d"]  # [x, y, z, x_size, y_size, z_size, yaw]
            velocity = pred_instance["velocity"]  # [vx, vy]
            instance_id = pred_instance["instance_id_3d"]
            bbox_label_id = pred_instance["bbox_label_3d"]
            score = pred_instance["bbox_score_3d"]
            timestamp = pseudo_label_info.get("timestamp", 0)
            # Convert label_id to label_name
            label = label_id_to_name.get(bbox_label_id, "car")

            if instance_id not in instance_ids_dict:
                instance_ids_dict[instance_id] = label_ids_count[label]
                label_ids_count[label] += 1

            # Prepare T4Box3D arguments
            position = [bbox_3d[0], bbox_3d[1], bbox_3d[2]]
            rotation = Quaternion(axis=[0, 0, 1], radians=bbox_3d[6])
            shape = Shape(shape_type=ShapeType.BOUNDING_BOX, size=(bbox_3d[4], bbox_3d[3], bbox_3d[5]))
            velocity3d = (velocity[0], velocity[1], 0.0)
            bbox = T4Box3D(
                unix_time=int(timestamp),
                frame_id="base_link",
                semantic_label=SemanticLabel(label),
                position=position,
                rotation=rotation,
                shape=shape,
                velocity=velocity3d,
                confidence=score,
                uuid=instance_id,
            )

            # Use ego2global matrix directly, treat lidar and ego as identical
            ego2global = np.array(pseudo_label_info["ego2global"])  # 4x4 matrix
            bbox_global = box_lidar_2_global(
                bbox,
                ego2global,
            )
            # Use T4Box3D's position, size, and rotation.q
            pos = bbox_global.position
            size = bbox_global.size
            rot = bbox_global.rotation.q.tolist()  # [w, x, y, z]
            scenes_anno_dict[scene_id].append(
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
                        "cx": float(pos[0]),
                        "cy": float(pos[1]),
                        "cz": float(pos[2]),
                        "w": float(size[0]),
                        "l": float(size[1]),
                        "h": float(size[2]),
                        "quaternion": {
                            "w": float(rot[0]),
                            "x": float(rot[1]),
                            "y": float(rot[2]),
                            "z": float(rot[3]),
                        },
                    },
                }
            )
    return scenes_anno_dict


def save_deepen_json(scenes_anno_dict, output_dir):
    """Save annotations in Deepen format."""
    for scene_name, scene_anno_dict in scenes_anno_dict.items():
        file_path = Path(output_dir) / f"{scene_name}.json"
        print(f"Generate deepen format json: {file_path}")
        deepen_anno_json = {"labels": scene_anno_dict}
        with open(file_path, "w") as f:
            json.dump(deepen_anno_json, f, indent=4)


def save_segment_ai_json(scenes_anno_dict, output_dir):
    """Save annotations in Segment.ai format (placeholder for future implementation)."""
    raise NotImplementedError("Segment.ai format is not yet supported")


def convert_pseudo_to_annotation_format(pseudo_label_path, output_dir, output_format, dataset_id=None):
    """Convert pseudo label to specified annotation tool format.
    
    Args:
        pseudo_label_path: Path to pseudo label pickle file
        output_dir: Output directory for annotation files
        output_format: Output format ('deepen' or 'segment.ai')
        dataset_id: Dataset ID to use for all annotations (overrides scene_name/sample_idx)
    """
    pseudo_label_infos, metainfo = load_pseudo_label(pseudo_label_path)
    scenes_anno_dict = get_scenes_anno_dict(pseudo_label_infos, metainfo, dataset_id=dataset_id)
    
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
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=False,
        default="Pseudo_DB_J6Gen2_v2_1_Shiojiri_kids_x2_dev_b3902d62-9777-496d-846b-39c3db7b9dcf_2025-04-09_11-34-42_11-35-02",
        help="Dataset ID to use for all annotations (overrides scene_name/sample_idx)",
    )
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    convert_pseudo_to_annotation_format(
        pseudo_label_path=args.input,
        output_dir=args.output_dir,
        output_format=args.output_format,
        dataset_id=args.dataset_id
    )


if __name__ == "__main__":
    main()
