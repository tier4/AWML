from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


def _load_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


@dataclass
class AWMLInfo:
    """Container for inference results stored in info.pkl"""

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
        """
        Load info.pkl, group by dataset name in a single pass,
        and return a list of AWMLInfo objects.
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
        return list(self.metainfo["classes"])

    @property
    def sorted_data_list(self) -> list[dict[str, Any]]:
        return self._sorted_data_list


@dataclass
class AWML3DInfo(AWMLInfo):
    """Container for 3D object inference results stored in info.pkl"""

    @classmethod
    def load(cls, info_path: str | Path) -> list["AWML3DInfo"]:
        """
        Load info.pkl, group by dataset name in a single pass,
        and return a list of AWML3DInfo objects.
        """
        # This is a type-safe way to call the parent's load method
        # and get a list of AWML3DInfo instances.
        return super().load(info_path)  # type: ignore

    def iter_frames(self) -> Iterable[dict[str, Any]]:
        for info in self.sorted_data_list:
            yield info
