from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_info(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


@dataclass
class AWMLInfo:
    """Container for inference results stored in info.pkl"""

    data_list: List[Dict[str, Any]] = field(default_factory=list)
    metainfo: Dict[str, Any] = field(default_factory=dict)
    dataset_id: Optional[str] = None

    def __post_init__(self) -> None:
        self._sorted_data_list: List[Dict[str, Any]] = sorted(
            self.data_list,
            key=lambda info: info.get("timestamp", 0),
        )

    @classmethod
    def load(cls, info_path: str | Path, dataset_id: Optional[str] = None) -> "AWMLInfo":
        path = Path(info_path)
        if not path.exists():
            raise FileNotFoundError(f"Info file not found: {path}")

        info_data = _load_info(path)
        if "data_list" in info_data:
            data_list = info_data["data_list"]
        else:
            data_list = info_data.get("infos", [])

        metainfo = info_data.get("metainfo", {})
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
    """Container for 3D object inference results stored in info.pkl"""

    def iter_frames(self) -> Iterable[Dict[str, Any]]:
        for info in self.sorted_data_list:
            yield info
