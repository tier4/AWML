from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from pyquaternion import Quaternion
from t4_devkit.dataclass import Box3D as T4Box3D
from t4_devkit.dataclass import SemanticLabel, Shape, ShapeType

from ..dataclass.awml_info import AWML3DInfo


@dataclass
class DeepenQuaternionFields:
    w: float
    x: float
    y: float
    z: float


@dataclass
class Deepen3DBBoxFields:
    cx: float
    cy: float
    cz: float
    w: float
    l: float
    h: float
    quaternion: DeepenQuaternionFields


@dataclass
class DeepenAnnotationFields:
    dataset_id: Optional[str]
    file_id: str
    label_category_id: str
    label_id: str
    instance_id: str
    label_type: str
    attributes: dict[str, str]
    labeller_email: str
    sensor_id: str
    three_d_bbox: Deepen3DBBoxFields


def _box_lidar_to_global(lidar_box: T4Box3D, ego2global: np.ndarray) -> T4Box3D:
    global_box: T4Box3D = copy.deepcopy(lidar_box)
    rotation = ego2global[:3, :3]
    translation = ego2global[:3, 3]
    global_box.rotate(Quaternion(matrix=rotation, rtol=1e-5, atol=1e-7))
    global_box.translate(translation)
    return global_box


@dataclass
class AnnotationToolDataset:
    """Base class for annotation datasets generated from AWML pseudo labels."""

    t4_dataset_name: str
    ann_tool_id: str
    ann_tool_file_path: Path
    scene_annotations: dict[str, list[Dict]] = field(default_factory=dict)

    @classmethod
    def create_from_info(
        cls,
        info: "AWML3DInfo",
        t4_dataset_name: str,
        ann_tool_id: str,
        ann_tool_file_path: Path,
    ) -> "AnnotationToolDataset":  # pragma: no cover - abstract
        """Factory method to create and save the dataset."""
        raise NotImplementedError


@dataclass
class DeepenDataset(AnnotationToolDataset):
    @classmethod
    def create_from_info(
        cls,
        info: "AWML3DInfo",
        t4_dataset_name: str,
        ann_tool_id: str,
        output_dir: Path,
    ) -> "DeepenDataset":
        """Factory method to create and save the Deepen dataset."""
        scene_annotations = cls._build_scene_annotations(info, ann_tool_id)
        output_filename = f"Pseudo_{t4_dataset_name}.json"
        ann_tool_file_path = output_dir / output_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        instance = cls(
            scene_annotations=scene_annotations,
            t4_dataset_name=t4_dataset_name,
            ann_tool_id=ann_tool_id,
            ann_tool_file_path=ann_tool_file_path,
        )

        deepen_payload = {"labels": scene_annotations[ann_tool_id]}
        with ann_tool_file_path.open("w") as handle:
            json.dump(deepen_payload, handle, indent=4)

        instance.ann_tool_file_path = ann_tool_file_path
        return instance

    @staticmethod
    def _build_scene_annotations(info: AWML3DInfo, tool_id: str) -> dict[str, list[Dict]]:
        scenes_anno_dict: dict[str, list[Dict]] = defaultdict(list)
        label_id_to_name = {label_id: class_name for label_id, class_name in enumerate(info.classes)}

        instance_ids_dict: dict[str, int] = {}
        label_ids_count: defaultdict[str, int] = defaultdict(lambda: 1)

        for idx, pseudo_label_info in enumerate(info.iter_frames()):
            file_id = f"{idx}.pcd"
            pred_instances = pseudo_label_info.get("pred_instances_3d", [])

            ego2global = np.array(pseudo_label_info["ego2global"])

            for pred_instance in pred_instances:
                bbox_3d = pred_instance["bbox_3d"]
                velocity = pred_instance["velocity"]
                instance_uuid = pred_instance["instance_id_3d"]
                bbox_label_id = pred_instance["bbox_label_3d"]
                score = pred_instance["bbox_score_3d"]
                timestamp = pseudo_label_info.get("timestamp", 0)
                label_name = label_id_to_name.get(bbox_label_id, "car")

                if instance_uuid not in instance_ids_dict:
                    instance_ids_dict[instance_uuid] = label_ids_count[label_name]
                    label_ids_count[label_name] += 1

                position = [bbox_3d[0], bbox_3d[1], bbox_3d[2]]
                rotation = Quaternion(axis=[0, 0, 1], radians=bbox_3d[6])
                shape = Shape(
                    shape_type=ShapeType.BOUNDING_BOX,
                    size=(bbox_3d[4], bbox_3d[3], bbox_3d[5]),
                )
                velocity3d = (velocity[0], velocity[1], 0.0)
                box = T4Box3D(
                    unix_time=int(timestamp),
                    frame_id="base_link",
                    semantic_label=SemanticLabel(label_name),
                    position=position,
                    rotation=rotation,
                    shape=shape,
                    velocity=velocity3d,
                    confidence=score,
                    uuid=instance_uuid,
                )

                bbox_global = _box_lidar_to_global(box, ego2global)
                pos = bbox_global.position
                size = bbox_global.size
                quat = bbox_global.rotation.q.tolist()

                annotation_fields = DeepenAnnotationFields(
                    dataset_id=tool_id,
                    file_id=file_id,
                    label_category_id=label_name,
                    label_id=f"{label_name}:{instance_ids_dict[instance_uuid]}",
                    instance_id=instance_uuid,
                    label_type="3d_bbox",
                    attributes={"pseudo-label": "auto-labeled"},
                    labeller_email="pseudo-label@AWML",
                    sensor_id="lidar",
                    three_d_bbox=Deepen3DBBoxFields(
                        cx=float(pos[0]),
                        cy=float(pos[1]),
                        cz=float(pos[2]),
                        w=float(size[0]),
                        l=float(size[1]),
                        h=float(size[2]),
                        quaternion=DeepenQuaternionFields(
                            w=float(quat[0]),
                            x=float(quat[1]),
                            y=float(quat[2]),
                            z=float(quat[3]),
                        ),
                    ),
                )

                scenes_anno_dict[tool_id].append(asdict(annotation_fields))
        return scenes_anno_dict


@dataclass
class SegmentAIDataset(AnnotationToolDataset):
    @classmethod
    def create_from_info(
        cls,
        info: "AWML3DInfo",
        output_dir: Path,
        t4_dataset_name: str,
        ann_tool_id: str,
        ann_tool_file_path: Path,
    ) -> "SegmentAIDataset":
        raise NotImplementedError("Segment.ai format is not yet supported")
