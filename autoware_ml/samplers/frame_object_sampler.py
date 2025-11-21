from abc import ABC, abstractmethod
from typing import List, Tuple

from mmengine.registry import DATA_SAMPLERS

SAMPLE_CLASS_NAME_KEY = "SAMPLE_CLASS_NAME"
FILTER_CLASS_LABELS = "FILTERED"


@DATA_SAMPLERS.register_module()
class ObjectSampler(ABC):
    """Base class for sampling strategies for objects."""

    @property
    def sampler_category_name(self) -> str:
        return ""

    @abstractmethod
    def sample(self, frame_ann_info: dict):
        pass


@DATA_SAMPLERS.register_module()
class FrameObjectSampler:
    """Base class for sampling strategies."""

    def __init__(self, object_samplers: List[ObjectSampler]):
        self.object_samplers = object_samplers

    @property
    def sampler_category_names(self) -> List[str]:
        return [sampler.sampler_category_name for sampler in self.object_samplers if sampler.sampler_category_name]

    def sample(self, frame_ann_info: dict):

        # Assign empty sample class name
        if SAMPLE_CLASS_NAME_KEY not in frame_ann_info:
            frame_ann_info[SAMPLE_CLASS_NAME_KEY] = ["" for _ in range(len(frame_ann_info["gt_labels_3d"]))]

        # Note that ObJectSampler modifies frame_ann_info in place, and it affects subsequent samplers.
        # For example, if an object is sampled by the first sampler, it may be sampled by the second sampler again
        # by assinging to a new class name. Please to make sure that the samplers are compatible with each other.
        for object_sampler in self.object_samplers:
            object_sampler.sample(frame_ann_info)


@DATA_SAMPLERS.register_module()
class ObjectBEVDistanceSampler(ObjectSampler):
    """Sampling strategy for neaerer objects."""

    def __init__(self, bev_distance_thresholds: Tuple[float, float, float, float]):
        super().__init__()
        self.bev_distance_thresholds = bev_distance_thresholds

    def sample(self, frame_ann_info: dict):
        nearer_bbox_in_ranges = frame_ann_info["gt_bboxes_3d"].in_range_bev(self.bev_distance_thresholds)
        for index, nearer_bbox_mask in enumerate(nearer_bbox_in_ranges):
            if nearer_bbox_mask:
                continue

            frame_ann_info[SAMPLE_CLASS_NAME_KEY][index] = FILTER_CLASS_LABELS


@DATA_SAMPLERS.register_module()
class NearerLowPedestrianObjectSampler(ObjectSampler):
    """Sampling strategy for nearer low pedestrian category."""

    def __init__(self, height_threshold: float, bev_distance_thresholds: Tuple[float, float, float, float]):
        super().__init__()
        self.height_threshold = height_threshold
        self.bev_distance_thresholds = bev_distance_thresholds
        self.mapping_category_name = "pedestrian"
        self.sampler_category_name = "low_pedestrian"

    @property
    def sampler_category_name(self) -> str:
        return self.sampler_category_name

    def sample(self, frame_ann_info: dict):
        # Implement the sampling logic for low pedestrian category
        gt_bbox_height_mask = frame_ann_info["gt_bboxes_3d"].height < self.height_threshold
        labels = frame_ann_info["gt_labels_3d"]
        for index, (label, gt_bbox_height_mask) in enumerate(zip(labels, gt_bbox_height_mask)):
            if (
                frame_ann_info[SAMPLE_CLASS_NAME_KEY][index] == FILTER_CLASS_LABELS
                or label != self.mapping_category_name
                or not gt_bbox_height_mask
            ):
                continue

            frame_ann_info[SAMPLE_CLASS_NAME_KEY][index] = self.sampler_category_name
