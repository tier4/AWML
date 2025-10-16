from __future__ import annotations

from dataclasses import dataclass
from typing import List

from t4_devkit.dataclass import Box2D

from tools.analysis_3d.data_classes import DetectionBox, SampleData


@dataclass(frozen=True)
class Detection2DBox(DetectionBox):
    """2D boxes from detection."""

    box: Box2D


@dataclass(frozen=True)
class SampleData2D(SampleData):
    """Dataclass to save data for a sample, for example, 2D bounding boxes."""

    sample_token: str
    detection_boxes: List[Detection2DBox]

    @classmethod
    def create_sample_data(
        cls,
        sample_token: str,
        boxes: List[Box2D],
    ) -> SampleData2D:
        """
        Create a SampleData2D given the params.
        :param sample_token: Sample token to represent a sample (lidar frame).
        :param boxes: List of 2D bounding boxes for the given sample token.
        """
        detection_2d_boxes = [Detection2DBox(box=box, attrs=box.semantic_label.attributes) for box in boxes]

        return cls(
            sample_token=sample_token,
            detection_boxes=detection_2d_boxes,
        )
