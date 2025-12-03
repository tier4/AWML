from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pyquaternion import Quaternion
from t4_devkit.dataclass import Box3D as T4Box3D
from t4_devkit.dataclass import SemanticLabel, Shape, ShapeType


def _load_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _box_ego_to_global(ego_box: T4Box3D, ego2global: np.ndarray) -> T4Box3D:
    """Convert a T4Box3D from the ego frame to the global frame.

    Args:
        ego_box (T4Box3D): A T4Box3D object in the ego frame.
        ego2global (np.ndarray): The transformation matrix from the ego frame to the global frame.

    Returns:
        T4Box3D: The T4Box3D object converted to the global frame.
    """
    global_box: T4Box3D = copy.deepcopy(ego_box)
    rotation = ego2global[:3, :3]
    translation = ego2global[:3, 3]
    global_box.rotate(Quaternion(matrix=rotation, rtol=1e-5, atol=1e-7))
    global_box.translate(translation)
    return global_box


@dataclass
class AWMLInfo:
    """Container for inference results stored in info.pkl

    Attributes:
        t4_dataset_name (str): The name of the T4 dataset.
        data_list (list[dict[str, Any]]): A list of dictionaries, each containing data for a single frame.
        metainfo (dict[str, Any]): A dictionary containing metadata, such as class names.
    """

    t4_dataset_name: str
    data_list: list[dict[str, Any]] = field(default_factory=list)
    metainfo: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._sorted_data_list: list[dict[str, Any]] = sorted(
            self.data_list,
            key=lambda info: info["timestamp"],
        )

    @classmethod
    def load(cls, info_path: str | Path) -> list["AWMLInfo"]:
        """Load info.pkl, group by dataset name, and return a list of AWMLInfo objects.

        Args:
            info_path (str | Path): The path to the info.pkl file.

        Raises:
            FileNotFoundError: If the specified path does not exist.

        Returns:
            list["AWMLInfo"]: A list of AWMLInfo objects grouped by each dataset.
        """
        path = Path(info_path)
        if not path.exists():
            raise FileNotFoundError(f"Info file not found: {path}")

        info_data = _load_info(path)
        data_list = info_data["data_list"]
        metainfo = info_data["metainfo"]

        grouped_data: dict[str, list[dict]] = {}
        for record in data_list:
            t4_dataset_name: str = record["scene_name"]
            grouped_data.setdefault(t4_dataset_name, []).append(record)

        return [
            cls(
                data_list=records,
                metainfo=metainfo,
                t4_dataset_name=name,
            )
            for name, records in grouped_data.items()
        ]

    @property
    def classes(self) -> list[str]:
        """Return the list of class names used in the dataset."""
        return list(self.metainfo["classes"])

    @property
    def sorted_data_list(self) -> list[dict[str, Any]]:
        """Return the data list sorted by timestamp."""
        return self._sorted_data_list


@dataclass
class AWML3DInfo(AWMLInfo):
    """Container for 3D object inference results stored in info.pkl"""

    @classmethod
    def load(cls, info_path: str | Path) -> list["AWML3DInfo"]:
        """Load info.pkl, group by dataset name, and return a list of AWML3DInfo objects.

        Args:
            info_path (str | Path): The path to the info.pkl file.

        Returns:
            list["AWML3DInfo"]: A list of AWML3DInfo objects grouped by each dataset.
        """
        # This is a type-safe way to call the parent's load method
        # and get a list of AWML3DInfo instances.
        return super().load(info_path)  # type: ignore

    def iter_frames(self) -> Iterable[dict[str, Any]]:
        """Iterator that yields sorted frame information.

        Yields:
            dict[str, Any]: Information for a single frame.
        """
        for info in self.sorted_data_list:
            yield info

    def iter_t4boxes_per_frame(self, global_frame: bool = False) -> Iterable[list[T4Box3D]]:
        """Generator that yields a list of T4Box3D objects for each frame.

        Args:
            global_frame (bool): If True, converts boxes to the global coordinate system. Defaults to False.

        Yields:
            Iterable[list[T4Box3D]]: A list of T4Box3D objects for one frame.
        """
        label_id_to_name = {label_id: class_name for label_id, class_name in enumerate(self.classes)}

        for frame_info in self.iter_frames():
            boxes_in_frame: list[T4Box3D] = []
            ego2global = np.array(frame_info["ego2global"])

            for pred_instance in frame_info["pred_instances_3d"]:
                bbox_3d = pred_instance["bbox_3d"]
                velocity = pred_instance["velocity"]
                label_name = label_id_to_name[pred_instance["bbox_label_3d"]]

                box_ego = T4Box3D(
                    unix_time=int(frame_info["timestamp"]),
                    frame_id="base_link",
                    semantic_label=SemanticLabel(label_name),
                    position=[bbox_3d[0], bbox_3d[1], bbox_3d[2]],
                    rotation=Quaternion(axis=[0, 0, 1], radians=bbox_3d[6]),
                    shape=Shape(
                        shape_type=ShapeType.BOUNDING_BOX,
                        size=(bbox_3d[4], bbox_3d[3], bbox_3d[5]),
                    ),
                    velocity=(velocity[0], velocity[1], 0.0),
                    confidence=pred_instance["bbox_score_3d"],
                    uuid=pred_instance["instance_id_3d"],
                )

                if global_frame:
                    box_global = _box_ego_to_global(box_ego, ego2global)
                    boxes_in_frame.append(box_global)
                else:
                    boxes_in_frame.append(box_ego)
            yield boxes_in_frame
