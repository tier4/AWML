import gc
import math
from os import path as osp
from typing import Dict, List, Optional

import numpy as np
from mmdet3d.datasets import NuScenesDataset
from mmengine.logging import print_log
from mmengine.registry import DATA_SAMPLERS, DATASETS

from autoware_ml.samplers.frame_object_sampler import FILTER_CLASS_LABELS, SAMPLE_CLASS_NAME_KEY, FrameObjectSampler


@DATASETS.register_module()
class T4Dataset(NuScenesDataset):
    """T4Dataset Dataset base class

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
        T4Dataset.METAINFO = metainfo
        self.frame_object_sampler: Optional[FrameObjectSampler] = (
            DATA_SAMPLERS.build(frame_object_sampler) if frame_object_sampler is not None else None
        )
        if self.frame_object_sampler is not None:
            class_names += self.frame_object_sampler.category_names

        self.class_names = class_names
        self.valid_class_name_ins = {class_name: 0 for class_name in class_names}

        # Repeat Sampling Factor variables
        # Number of frames for each category that contains at least one of the category
        self.category_frame_number = {class_name: 0 for class_name in class_names}
        self.repeat_sampling_factor = repeat_sampling_factor

        # Save annotation info for all frames to compute frame repeat sampling factor
        self.ann_info = []
        super().__init__(use_valid_flag=use_valid_flag, **kwargs)

        # Repeat sampling factor computation
        # Compute category frame fraction
        self.category_frame_number = {
            class_name: value / len(self) for class_name, value in self.category_frame_number.items()
        }
        # Total valid bbox
        total_bboxes = sum(self.valid_class_name_ins.values())

        # Compute bbox fraction
        self.valid_class_bbox_fraction = {
            class_name: class_bbox_num / total_bboxes
            for class_name, class_bbox_num in self.valid_class_name_ins.items()
        }

        self.category_fraction_factor = self._compute_category_faction_factor()
        self.frame_weights = self._compute_frame_repeat_sampling_factor()

        # Print dataset statistics and clean up
        self.print_dataset_statistics()
        self.ann_info.clear()
        gc.collect()

    def print_dataset_statistics(self) -> None:
        """Print dataset statistics."""
        print_log(f"Valid dataset instances: {self.valid_class_name_ins}", logger="current")
        print_log(f"Category frame fraction: {self.category_frame_number}", logger="current")
        print_log(f"Valid bbox fraction: {self.valid_class_bbox_fraction}", logger="current")
        print_log(f"Category fraction factor: {self.category_fraction_factor}", logger="current")
        print_log(f"First 10 dataset frame weights: {self.frame_weights[:10]}", logger="current")

    def filter_data(self) -> List[dict]:
        """
        Overriding from superclass.

        Filter annotations according to filter_cfg. Defaults return all
        ``data_list``in Superclass.

        If some ``data_list`` could be filtered according to specific logic,
        the subclass should override this method.

        Returns:
            List[dict]: Filtered results.
        """
        if not self.filter_cfg:
            return self.data_list
        filtered_data_list = []
        for entry in self.data_list:
            if self.filter_cfg.get("filter_frames_with_missing_image", False) and not all(
                [x["img_path"] and osp.exists(x["img_path"]) for x in entry["images"].values()]
            ):
                continue
            filtered_data_list.append(entry)

        if len(filtered_data_list) != len(self.data_list):
            print_log(
                f"Filtered {len(self.data_list)-len(filtered_data_list)}/{len(self.data_list)} frames without images.",
                logger="current",
            )

        return filtered_data_list

    def _compute_category_faction_factor(self) -> Dict[str, float]:
        """Compute category fraction factor used for repeat sampling factor computation."""
        category_fraction_factor = {
            class_name: max(
                1,
                math.sqrt(
                    self.repeat_sampling_factor
                    / math.sqrt((number_frame * self.valid_class_bbox_fraction[class_name]))
                ),
            )
            for class_name, number_frame in self.category_frame_number.items()
        }
        return category_fraction_factor

    def _compute_frame_repeat_sampling_factor(self) -> List[float]:
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
                    frame_repeat_sampling_factor, self.category_fraction_factor[class_name]
                )
            frame_weights.append(frame_repeat_sampling_factor)

        return frame_weights

    def _filter_with_mask(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos.

        Returns:
            dict: Annotations after filtering.
        """
        filtered_annotations = {}
        if self.use_valid_flag:
            filter_mask = ann_info["bbox_3d_isvalid"]
        else:
            # For safety reason, we should check if there's > -1 to take all valid ground truths
            # only
            # There's _remove_dontcare() in the implementation of both KittiDataset and
            # WaymoDataset, but no in NuScenesDataset
            filter_mask = (ann_info["num_lidar_pts"] > 0) & (ann_info["gt_labels_3d"] > -1)

        for key in ann_info.keys():
            if key != "instances":
                filtered_annotations[key] = ann_info[key][filter_mask]
            else:
                filtered_annotations[key] = ann_info[key]
        return filtered_annotations

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

        valid_bbox_categories = {class_name: 0 for class_name in self.class_names}
        for label, sample_class_name in zip(ann_info["gt_labels_3d"], ann_info[SAMPLE_CLASS_NAME_KEY]):

            if sample_class_name == FILTER_CLASS_LABELS:
                continue

            class_name = self.class_names[label] if not sample_class_name else sample_class_name
            self.valid_class_name_ins[class_name] += 1
            # Set to 1 if a category exists in this frame
            valid_bbox_categories[class_name] = 1

        # Sum up category fraction for this frame
        for class_name in self.class_names:
            self.category_frame_number[class_name] += valid_bbox_categories[class_name]

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

        use_lidar = self.modality["use_lidar"]
        use_camera = self.modality["use_camera"]
        self.modality["use_lidar"] = False
        self.modality["use_camera"] = False

        info = super().parse_data_info(info)
        self.modality["use_lidar"] = use_lidar
        self.modality["use_camera"] = use_camera

        # modified from https://github.com/open-mmlab/mmdetection3d/blob/v1.2.0/mmdet3d/datasets/det3d_dataset.py#L279-L296
        if self.modality["use_lidar"]:
            info["lidar_points"]["lidar_path"] = osp.join(
                self.data_prefix.get("pts", ""), info["lidar_points"]["lidar_path"]
            )
            info["num_pts_feats"] = info["lidar_points"]["num_pts_feats"]
            info["lidar_path"] = info["lidar_points"]["lidar_path"]
            if "lidar_sweeps" in info:
                for sweep in info["lidar_sweeps"]:
                    # NOTE: modified to avoid hard-coded processes for nuscenes dataset
                    file_suffix = sweep["lidar_points"]["lidar_path"]
                    # -----------------------------------------------
                    if "samples" in sweep["lidar_points"]["lidar_path"]:
                        sweep["lidar_points"]["lidar_path"] = osp.join(self.data_prefix["pts"], file_suffix)
                    else:
                        sweep["lidar_points"]["lidar_path"] = osp.join(self.data_prefix["sweeps"], file_suffix)

        if self.modality["use_camera"]:
            for cam_id, img_info in info["images"].items():
                if "img_path" in img_info:
                    if cam_id in self.data_prefix:
                        cam_prefix = self.data_prefix[cam_id]
                    else:
                        cam_prefix = self.data_prefix.get("img", "")
                    # If an image is invalid, then set img_info['img_path'] = None
                    if img_info["img_path"] is None:
                        img_info["img_path"] = None
                    else:
                        img_info["img_path"] = osp.join(
                            cam_prefix,
                            img_info["img_path"],
                        )

            if self.default_cam_key is not None:
                info["img_path"] = info["images"][self.default_cam_key]["img_path"]
                if "lidar2cam" in info["images"][self.default_cam_key]:
                    info["lidar2cam"] = np.array(info["images"][self.default_cam_key]["lidar2cam"])
                if "cam2img" in info["images"][self.default_cam_key]:
                    info["cam2img"] = np.array(info["images"][self.default_cam_key]["cam2img"])
                if "lidar2img" in info["images"][self.default_cam_key]:
                    info["lidar2img"] = np.array(info["images"][self.default_cam_key]["lidar2img"])
                else:
                    info["lidar2img"] = info["cam2img"] @ info["lidar2cam"]

        if "ann_info" in info:
            self.ann_info.append(info["ann_info"])
        return info
