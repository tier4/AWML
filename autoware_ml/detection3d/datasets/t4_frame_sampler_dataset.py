import gc
import math
from typing import Dict, List, Optional

import numpy as np
from mmengine.logging import print_log
from mmengine.registry import DATA_SAMPLERS, DATASETS

from autoware_ml.detection3d.datasets.t4dataset import T4Dataset
from autoware_ml.samplers.frame_object_sampler import FILTER_CLASS_LABELS, SAMPLE_CLASS_NAME_KEY, FrameObjectSampler


@DATASETS.register_module()
class T4FrameSamplerDataset(T4Dataset):
    """T4Dataset with FrameObjectSampler to sample objects in certain frames to support frame weighting.

    This dataset class extends NuScenesDataset to provide specialized functionality
    for T4 dataset processing with additional filtering and validation capabilities.

    Args:
        metainfo: Metadata information for the dataset.
        class_names: List of class names for object detection/classification.
        use_valid_flag (bool, optional): Whether to use validity flags for filtering
            annotations. Defaults to False.
        **kwargs: Additional keyword arguments passed to the parent NuScenesDataset.
        repeat_sampling_factor (float, optional): A threshold value to compute repeat sampling factor
        `max(1, sqrt(repeat_sampling_factor / frame_sampling_factor))` for every frame.
        special_class_sampling (Dict[str, Dict[str, Tuple[float, List[float, float]]]], optional): A dictionary
            specifying special classes and their sampling criteria.
            The key is the (original class name, special class name), and the value is a tuple containing
            the sampling factor and a list with height and distance thresholds. For example,
            to add a new class 'low_pedestrian' for pedestrians to repeat sampling factor lower than 1.5m
            height and within 10m from the ego vehicle, you can specify:
            {'pedestrian': {'low_pedestrian': (1.5, 10.0)}}.
    """

    def __init__(
        self,
        metainfo,
        class_names,
        use_valid_flag: bool = False,
        repeat_sampling_factor: Optional[float] = None,
        frame_object_sampler: Optional[dict] = None,
        **kwargs,
    ):
        # These have to be initialized before the parent class init since parse_ann_info and parse_data_info
        # are called in the parent class
        self.ann_info = []

        self.augmented_class_names = class_names
        if self.frame_object_sampler is not None:
            self.augmented_class_names += self.frame_object_sampler.sampler_category_names
        self.valid_class_name_ins = {class_name: 0 for class_name in class_names}

        self.frame_object_sampler: Optional[FrameObjectSampler] = (
            DATA_SAMPLERS.build(frame_object_sampler) if frame_object_sampler is not None else None
        )
        # Number of frames for each category that contains at least one of the category
        self.category_frame_numbers = {class_name: 0 for class_name in self.augmented_class_names}

        super().__init__(
            metainfo=metainfo,
            class_names=class_names,
            use_valid_flag=use_valid_flag,
            **kwargs,
        )

        self.repeat_sampling_factor = repeat_sampling_factor

        # Total valid bbox
        total_bboxes = sum(self.valid_class_name_ins.values())

        # Compute bbox fraction
        self.valid_class_bbox_fraction = {
            class_name: class_bbox_num / total_bboxes
            for class_name, class_bbox_num in self.valid_class_name_ins.items()
        }

        # Compute category frame fraction
        self.category_fraction_factors = self._compute_category_faction_factors()
        self.frame_weights = self._compute_frame_repeat_sampling_factors()

        # Print dataset statistics and clean up
        self.print_dataset_statistics()
        self.ann_info.clear()
        gc.collect()

    def print_dataset_statistics(self) -> None:
        """Print dataset statistics."""
        print_log(f"Valid dataset instances: {self.valid_class_name_ins}", logger="current")
        print_log(f"Category frame fraction: {self.category_frame_numbers}", logger="current")
        print_log(f"Valid bbox fraction: {self.valid_class_bbox_fraction}", logger="current")
        print_log(f"Category fraction factor: {self.category_fraction_factors}", logger="current")
        print_log(f"First 10 dataset frame weights: {self.frame_weights[:10]}", logger="current")

    def _compute_category_faction_factors(self) -> Dict[str, float]:
        """Compute category fraction factor used for repeat sampling factor computation."""
        category_fraction_factor = {
            class_name: max(
                1,
                math.sqrt(
                    self.repeat_sampling_factor
                    / math.sqrt((number_frame * self.valid_class_bbox_fraction[class_name]))
                ),
            )
            for class_name, number_frame in self.category_frame_numbers.items()
        }
        return category_fraction_factor

    def _compute_frame_repeat_sampling_factors(self) -> List[float]:
        """Compute repeat sampling factor for every frame in the dataset."""
        if self.repeat_sampling_factor is None:
            return [1.0] * len(self)

        frame_weights = []
        for ann_info in self.ann_info:
            valid_bbox_categories = ann_info["valid_bbox_categories"]
            frame_repeat_sampling_factor = 1
            for class_name, value in valid_bbox_categories.items():
                if not value:
                    continue

                frame_repeat_sampling_factor = max(
                    frame_repeat_sampling_factor, self.category_fraction_factors[class_name]
                )
            frame_weights.append(frame_repeat_sampling_factor)

        return frame_weights

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info=info)
        ann_info = self.frame_object_sampler.sample(ann_info) if self.frame_object_sampler else ann_info

        valid_bbox_categories = {class_name: 0 for class_name in self.augmented_class_names}
        for label, sample_class_name in zip(ann_info["gt_labels_3d"], ann_info[SAMPLE_CLASS_NAME_KEY]):

            if sample_class_name == FILTER_CLASS_LABELS:
                continue

            class_name = self.augmented_class_names[label] if not sample_class_name else sample_class_name
            self.valid_class_name_ins[class_name] += 1
            # Set to 1 if a category exists in this frame
            valid_bbox_categories[class_name] = 1

        # Sum up category fraction for this frame
        for class_name in self.augmented_class_names:
            self.category_frame_numbers[class_name] += valid_bbox_categories[class_name]

        ann_info["valid_bbox_categories"] = valid_bbox_categories
        return ann_info

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process the `instances` field to `ann_info` in training stage.
        This function is modified to avoid hard-coded processes for nuscenes dataset.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info = super().parse_data_info(info)
        if "ann_info" in info:
            self.ann_info.append(info["ann_info"])
        return info
