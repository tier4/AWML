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
    """Represents the quaternion for 3D bounding box orientation in Deepen format."""

    w: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Deepen3DBBoxFields:
    """Represents a 3D bounding box in the global coordinate system, formatted for Deepen."""

    cx: float
    cy: float
    cz: float
    w: float
    l: float
    h: float
    quaternion: DeepenQuaternionFields

    @classmethod
    def from_global_t4box(cls, global_box: T4Box3D) -> "Deepen3DBBoxFields":
        """Create Deepen3DBBoxFields from a T4Box3D in the global coordinate system.

        Args:
            global_box (T4Box3D): A T4Box3D object in the global coordinate system.

        Returns:
            Deepen3DBBoxFields: An instance of Deepen3DBBoxFields.
        """
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
    """Represents a single t4dataset annotation record in Deepen format."""

    dataset_id: str
    file_id: str
    label_category_id: str
    label_id: str
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
        """Assign a unique ID to the given UUID and label.

        If an ID has already been assigned to the UUID, it returns the existing one.
        A unique ID is in the format 'label_name:number', e.g., 'car:1'.

        Args:
            uuid (str): The instance UUID of the object.
            label (str): The semantic label of the object (e.g., 'car', 'pedestrian').

        Returns:
            str: The assigned unique ID for tracking.
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
        info: AWML3DInfo,
        t4_dataset_name: str,
        ann_tool_id: str,
    ) -> "AnnotationToolDataset":  # pragma: no cover - abstract
        """Factory method to create and save the dataset."""
        raise NotImplementedError


@dataclass(frozen=True)
class DeepenDataset(AnnotationToolDataset):
    """Represents a dataset in Deepen's JSON format, containing annotations for a scene."""

    scene_annotations: list[dict] = field(default_factory=list)

    @classmethod
    def create_from_info(
        cls,
        info: AWML3DInfo,
        t4_dataset_name: str,
        ann_tool_id: str,
        output_dir: Path,
    ) -> "DeepenDataset":
        """Create a DeepenDataset from an AWML3DInfo object and save it to a file.

        This class method orchestrates the conversion of raw AWML 3D data into the Deepen
        JSON format. It generates annotations, creates an instance of the class, and
        saves the resulting JSON file.

        Args:
            info (AWML3DInfo): The AWML3DInfo object containing the source data.
            t4_dataset_name (str): The name of the T4 dataset being processed.
            ann_tool_id (str): The unique identifier for this dataset in the annotation tool.
            output_dir (Path): The directory where the output JSON file will be saved.

        Returns:
            DeepenDataset: An instance of the DeepenDataset class, representing the converted data.
        """
        scene_annotations: list[dict] = cls._create_annotations_from_info(info, ann_tool_id)
        ann_tool_file_path: Path = output_dir / f"Pseudo_{t4_dataset_name}.json"
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
        """Serialize the scene annotations to a JSON file in the Deepen format.

        This method creates a payload dictionary with a "labels" key and writes the
        `scene_annotations` list to the file specified by `self.ann_tool_file_path`.
        The JSON is formatted with an indent of 4 for readability.
        """
        deepen_payload: dict[str, list[dict]] = {"labels": self.scene_annotations}
        with self.ann_tool_file_path.open("w") as handle:
            json.dump(deepen_payload, handle, indent=4)

    @staticmethod
    def _create_annotations_from_info(info: AWML3DInfo, tool_id: str) -> list[dict]:
        """Creates a list of Deepen-formatted annotations from AWML3DInfo.

        Args:
            info (AWML3DInfo): The AWML3DInfo object containing the raw dataset.
            tool_id (str): The dataset ID for the annotation tool.

        Returns:
            list[dict]: A list of dictionaries, where each represents a Deepen annotation.
        """
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
                    label_type="3d_bbox",
                    attributes={"pseudo-label": "auto-labeled"},
                    labeller_email="pseudo-label@AWML",
                    sensor_id="lidar",
                    three_d_bbox=Deepen3DBBoxFields.from_global_t4box(box_global),
                )

                annotations.append(asdict(annotation_fields))
        return annotations


@dataclass(frozen=True)
class SegmentsAIDataset(AnnotationToolDataset):
    @classmethod
    def create_from_info(
        cls,
        info: AWML3DInfo,
        output_dir: Path,
        t4_dataset_name: str,
        ann_tool_id: str,
    ) -> "SegmentsAIDataset":
        raise NotImplementedError("Segments.ai format is not yet supported")
