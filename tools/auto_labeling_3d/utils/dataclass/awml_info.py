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
    t4_dataset_name: str = ""

    def __post_init__(self) -> None:
        self._sorted_data_list: List[Dict[str, Any]] = sorted(
            self.data_list,
            key=lambda info: info.get("timestamp", 0),
        )

    @classmethod
    def load(cls, info_path: str | Path) -> List["AWMLInfo"]:
        """
        info.pklを読み込み、単一パスでデータセット名ごとにグループ化し、
        AWMLInfoオブジェクトのリストを返す。
        """
        path = Path(info_path)
        if not path.exists():
            raise FileNotFoundError(f"Info file not found: {path}")

        info_data = _load_info(path)
        data_list = info_data.get("data_list", info_data.get("infos", []))
        metainfo = info_data.get("metainfo", {})

        # 1. 単一パスでデータをグループ化
        grouped_data: Dict[str, List[Dict]] = {}
        for record in data_list:
            t4_dataset_name: str = record["scene_name"]
            grouped_data.setdefault(t4_dataset_name, []).append(record)

        # 2. グループ化したデータからAWMLInfoオブジェクトのリストを作成 (リスト内包表記)
        return [
            cls(
                data_list=records,
                metainfo=metainfo,
                t4_dataset_name=name,
            )
            for name, records in grouped_data.items()
        ]

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

    @classmethod
    def load(cls, info_path: str | Path) -> List["AWML3DInfo"]:
        """
        info.pklを読み込み、単一パスでデータセット名ごとにグループ化し、
        AWML3DInfoオブジェクトのリストを返す。
        """
        # This is a type-safe way to call the parent's load method
        # and get a list of AWML3DInfo instances.
        return super().load(info_path)  # type: ignore

    def iter_frames(self) -> Iterable[Dict[str, Any]]:
        for info in self.sorted_data_list:
            yield info
