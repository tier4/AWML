from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from t4_devkit.dataclass import Box3D as T4Box3D

from ..dataclass.awml_info import AWML3DInfo


@dataclass(frozen=True)
class DeepenQuaternionFields:
    w: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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

class DeepenUniqueId:
    """Manages the state for generating unique IDs for object tracking across frames."""

    def __init__(self) -> None:
        self._instance_uuid_to_unique_id: dict[str, str] = {}
        self._label_name_counts: defaultdict[str, int] = defaultdict(lambda: 1)

    def assign_id(self, uuid: str, label: str) -> str:
        """
        Assign a unique ID to the given UUID and label.
        If an ID has already been assigned to the UUID, it returns the existing one.
        A unique ID is in the format 'label_name:number', e.g., 'car:1'.
        """
        if uuid in self._instance_uuid_to_unique_id:
            return self._instance_uuid_to_unique_id[uuid]

        unique_id_num = self._label_name_counts[label]
        unique_label_id = f"{label}:{unique_id_num}"

        self._instance_uuid_to_unique_id[uuid] = unique_label_id
        self._label_name_counts[label] += 1

        return unique_label_id

@dataclass(frozen=True)
class AnnotationToolDataset:
    """Base class for annotation datasets generated from AWML pseudo labels."""

    t4_dataset_name: str
    ann_tool_id: str
    ann_tool_file_path: Path

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


@dataclass(frozen=True)
class DeepenDataset(AnnotationToolDataset):
    scene_annotations: list[dict] = field(default_factory=list)

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
        ann_tool_file_path = output_dir / f"Pseudo_{t4_dataset_name}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        instance = cls(
            scene_annotations=scene_annotations,
            t4_dataset_name=t4_dataset_name,
            ann_tool_id=ann_tool_id,
            ann_tool_file_path=ann_tool_file_path,
        )
        instance.dump_json()
        return instance

    def dump_json(self) -> None:
        """Dumps the scene annotations to a JSON file."""
        deepen_payload: dict[str, list[dict]] = {"labels": self.scene_annotations}
        with self.ann_tool_file_path.open("w") as handle:
            json.dump(deepen_payload, handle, indent=4)

    @staticmethod
    def _build_scene_annotations(info: AWML3DInfo, tool_id: str) -> list[dict]:
        annotations: list[dict] = []
        id_generator = DeepenUniqueId()

        for idx, boxes_in_frame_global in enumerate(info.iter_t4boxes_per_frame(global_frame=True)):
            file_id: str = f"{idx}.pcd"

            for box_global in boxes_in_frame_global:
                unique_label_id: str = id_generator.assign_id(box_global.uuid, str(box_global.semantic_label))

                annotation_fields = DeepenAnnotationFields(
                    dataset_id=tool_id,
                    file_id=file_id,
                    label_category_id=str(box_global.semantic_label),
                    label_id=unique_label_id,
                    instance_id=box_global.uuid,
                    label_type="3d_bbox",
                    attributes={"pseudo-label": "auto-labeled"},
                    labeller_email="pseudo-label@AWML",
                    sensor_id="lidar",
                    three_d_bbox=Deepen3DBBoxFields.from_global_t4box(box_global),
                )

                annotations.append(asdict(annotation_fields))
        return annotations


@dataclass(frozen=True)
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

