from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from mmengine.logging import print_log
from t4_devkit import Tier4 as t4
from t4_devkit.dataclass import Box3D


class DatasetSplitName(NamedTuple):
    """Represent a pair of dataset and a split name."""

    dataset_version: str
    split_name: str


@dataclass(frozen=True)
class Detection3DBox:
    """3D boxes from detection."""

    box: Box3D
    attrs: List[str]


@dataclass(frozen=True)
class LidarPoint:
    num_pts_feats: int
    lidar_path: str
    lidar2ego: npt.NDArray[np.float64]


@dataclass(frozen=True)
class LidarSweep:
    num_pts_feats: int
    lidar_path: str


@dataclass(frozen=True)
class SampleData:
    """Dataclass to save data for a sample, for example, 3D bounding boxes."""

    sample_token: str
    detection_3d_boxes: List[Detection3DBox]
    lidar_point: Optional[LidarPoint] = None  # Path to the lidar file
    lidar_sweeps: Optional[List[LidarSweep]] = None  # List of lidar sweeps

    def get_category_attr_counts(
        self,
        category_name: str,
        remapping_classes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        """
        Get total counts of every attribute for the selected category in this scenario.
        :param category_name: Selected category name.
        :param remapping_classes: Set if we want to aggregate the total counts after remapping
        categories.
        :return: A dict of {attribute name: total counts}.
        """
        category_attr_counts: Dict[str, int] = defaultdict(int)
        for detection_3d_box in self.detection_3d_boxes:
            box_category_name = detection_3d_box.box.semantic_label.name
            if remapping_classes is not None:
                # If no category found from the remapping, then it uses the original category name
                box_category_name = remapping_classes.get(box_category_name, box_category_name)

            if box_category_name == category_name:
                for attr_name in detection_3d_box.attrs:
                    category_attr_counts[attr_name] += 1

        return category_attr_counts

    def get_category_counts(
        self,
        remapping_classes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        """
        Get total counts of every category for every sample in this scenario.
        :param remapping_classes: Set if we want to aggregate the total counts after remapping
        categories.
        :return: A dict of {sample token: {category name: total counts}}.
        """
        category_counts: Dict[str, int] = defaultdict(int)
        for detection_3d_box in self.detection_3d_boxes:
            box_category_name = detection_3d_box.box.semantic_label.name
            if remapping_classes is not None:
                # If no category found from the remapping, then it uses the original category name
                box_category_name = remapping_classes.get(box_category_name, box_category_name)
            category_counts[box_category_name] += 1
        return category_counts

    @classmethod
    def create_sample_data(
        cls,
        sample_token: str,
        boxes: List[Box3D],
        lidar_point: Optional[LidarPoint] = None,
        lidar_sweeps: Optional[List[LidarSweep]] = None,
    ) -> SampleData:
        """
        Create a SampleData given the params.
        :param sample_token: Sample token to represent a sample (lidar frame).
        :param detection_3d_boxes: List of 3D bounding boxes for the given sample token.
        """
        detection_3d_boxes = [Detection3DBox(box=box, attrs=box.semantic_label.attributes) for box in boxes]

        return SampleData(
            sample_token=sample_token,
            detection_3d_boxes=detection_3d_boxes,
            lidar_sweeps=lidar_sweeps,
            lidar_point=lidar_point,
        )


@dataclass
class ScenarioData:
    """Data class to save data for a scenario, for example, a list of SampleData."""

    scene_token: str
    sample_data: Dict[str, SampleData] = field(default_factory=lambda: {})  # Sample token, SampleAnalysis

    def add_sample_data(self, sample_data: SampleData) -> None:
        """
        Add a SampleData to ScenarioData.
        :param sample_data: SampleData contains data for descripting a sample/lidar frame.
        """
        if sample_data.sample_token in self.sample_data:
            print_log(f"Found {sample_data.sample_token} in the data, replacing it...")
        self.sample_data[sample_data.sample_token] = sample_data

    def get_scenario_category_counts(
        self,
        remapping_classes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Get total counts of every category for every sample in this scenario.
        :param remapping_classes: Set if we want to aggregate the total counts after remapping
        categories.
        :return: A dict of {sample token: {category name: total counts}}.
        """
        scenario_category_counts: Dict[str, Dict[str, int]] = {}
        for sample_token, sample_data in self.sample_data.items():
            scenario_category_counts[sample_token] = sample_data.get_category_counts(
                remapping_classes=remapping_classes
            )
        return scenario_category_counts

    def get_scenario_category_attr_counts(
        self,
        category_name: str,
        remapping_classes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Get total counts of every attribute for the selected category in this scenario.
        :param category_name: Selected category name.
        :param remapping_classes: Set if we want to aggregate the total counts after remapping
        categories.
        :return: A dict of {sample token: {attribute name: total counts}}.
        """
        scenario_category_counts: Dict[str, Dict[str, int]] = {}
        for sample_token, sample_data in self.sample_data.items():
            scenario_category_counts[sample_token] = sample_data.get_category_attr_counts(
                category_name=category_name, remapping_classes=remapping_classes
            )
        return scenario_category_counts


@dataclass
class AnalysisData:
    """Data class to save data for an analysis, for example, a list of ScenarioData."""

    data_root_path: str
    dataset_version: str
    scenario_data: Dict[str, ScenarioData] = field(default_factory=lambda: {})  # Scene token, ScenarioAnalysis

    def add_scenario_data(self, scenario_data: ScenarioData) -> None:
        """
        Add a ScenarioData to AnalysisData.
        :param scenario_data: ScenarioData contains data for descripting a scenario (more than a
          sample/lidar frames).
        """
        if scenario_data.scene_token in self.scenario_data:
            print_log(f"Found {scenario_data.scene_token} in the data, replacing it...")
        self.scenario_data[scenario_data.scene_token] = scenario_data

    def aggregate_category_counts(
        self,
        remapping_classes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        """
        Get total counts of every category in this AnalysisData.
        :param remapping_classes: Set if we want to aggregate the total counts after remapping
        categories.
        :return: A dict of {category name: total counts}.
        """
        # {category_name: counts}
        total_category_counts = defaultdict(int)
        for scenario_data in self.scenario_data.values():
            scenario_category_counts: Dict[str, Dict[str, int]] = scenario_data.get_scenario_category_counts(
                remapping_classes=remapping_classes
            )
            for category_counts in scenario_category_counts.values():
                for name, counts in category_counts.items():
                    total_category_counts[name] += counts
        return total_category_counts

    def aggregate_category_attr_counts(
        self,
        category_name: str,
        remapping_classes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        """
        Get total counts of every attribute for the selected category in this AnalysisData.
        :param category_name: Selected category name.
        :param remapping_classes: Set if we want to aggregate the total counts after remapping
        categories.
        :return: A dict of {attribute name: total counts}.
        """
        # {category_name: counts}
        total_category_counts = defaultdict(int)
        for scenario_data in self.scenario_data.values():
            scenario_category_counts: Dict[str, Dict[str, int]] = scenario_data.get_scenario_category_attr_counts(
                category_name=category_name, remapping_classes=remapping_classes
            )

            for category_counts in scenario_category_counts.values():
                for name, counts in category_counts.items():
                    total_category_counts[name] += counts
        return total_category_counts
