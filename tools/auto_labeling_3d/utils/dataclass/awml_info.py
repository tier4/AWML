from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_pseudo_label(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


@dataclass
class AWMLInfo:
    """Container for pseudo label information used across annotation exporters."""

    data_list: List[Dict[str, Any]] = field(default_factory=list)
    metainfo: Dict[str, Any] = field(default_factory=dict)
    dataset_id: Optional[str] = None

    def __post_init__(self) -> None:
        self._sorted_data_list: List[Dict[str, Any]] = sorted(
            self.data_list,
            key=lambda info: info.get("timestamp", 0),
        )

    @classmethod
    def load(cls, pseudo_label_path: str | Path, dataset_id: Optional[str] = None) -> "AWMLInfo":
        path = Path(pseudo_label_path)
        if not path.exists():
            raise FileNotFoundError(f"Pseudo label file not found: {path}")

        pseudo_label_data = _load_pseudo_label(path)
        if "data_list" in pseudo_label_data:
            data_list = pseudo_label_data["data_list"]
        else:
            data_list = pseudo_label_data.get("infos", [])

        metainfo = pseudo_label_data.get("metainfo", {})
        return cls(data_list=data_list, metainfo=metainfo, dataset_id=dataset_id)

    @property
    def classes(self) -> List[str]:
        return list(self.metainfo.get("classes", ["car", "pedestrian", "bicycle"]))

    @property
    def sorted_data_list(self) -> List[Dict[str, Any]]:
        return self._sorted_data_list

    def get_label_name(self, label_id: int, default: str = "car") -> str:
        classes = self.classes
        if 0 <= label_id < len(classes):
            return classes[label_id]
        return default


@dataclass
class AWML3DInfo(AWMLInfo):
    """3D specific pseudo label container."""

    def iter_frames(self) -> Iterable[Dict[str, Any]]:
        for info in self.sorted_data_list:
            yield info
