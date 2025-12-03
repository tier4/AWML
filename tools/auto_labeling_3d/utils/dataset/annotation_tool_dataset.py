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

    @classmethod
    def from_global_t4box(cls, global_box: T4Box3D) -> "Deepen3DBBoxFields":
        """Create Deepen3DBBoxFields from a T4Box3D in the global coordinate system."""
        pos = global_box.position
        size = global_box.size
        quat = global_box.rotation.q

        return cls(
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
        )


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


def _box_ego_to_global(lidar_box: T4Box3D, ego2global: np.ndarray) -> T4Box3D:
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

        # --- State management for generating unique IDs ---
        instance_uuid_to_unique_id: dict[str, str] = {}
        label_name_counts: defaultdict[str, int] = defaultdict(lambda: 1)
        # ---

        # Use zip to get both frame info (for ego2global) and the list of T4Box3D objects
        for idx, (frame_info, boxes_in_frame) in enumerate(zip(info.iter_frames(), info.iter_t4boxes_per_frame())):
            file_id = f"{idx}.pcd"
            ego2global = np.array(frame_info["ego2global"])

            for box_ego in boxes_in_frame:
                instance_uuid = box_ego.uuid
                label_name = str(box_ego.semantic_label)

                # Generate a unique ID like "car:1", "pedestrian:3"
                if instance_uuid not in instance_uuid_to_unique_id:
                    unique_id_num = label_name_counts[label_name]
                    instance_uuid_to_unique_id[instance_uuid] = f"{label_name}:{unique_id_num}"
                    label_name_counts[label_name] += 1
                unique_label_id = instance_uuid_to_unique_id[instance_uuid]

                # Transform box to global coordinate system
                box_global = _box_ego_to_global(box_ego, ego2global)

                # Create Deepen fields from the global box
                three_d_bbox = Deepen3DBBoxFields.from_global_t4box(box_global)

                annotation_fields = DeepenAnnotationFields(
                    dataset_id=tool_id,
                    file_id=file_id,
                    label_category_id=label_name,
                    label_id=unique_label_id,
                    instance_id=instance_uuid,
                    label_type="3d_bbox",
                    attributes={"pseudo-label": "auto-labeled"},
                    labeller_email="pseudo-label@AWML",
                    sensor_id="lidar",
                    three_d_bbox=three_d_bbox,
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
