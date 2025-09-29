import gc
import math
import pickle
from os import path as osp
from typing import List, Optional

import numpy as np
from mmdet3d.datasets import NuScenesDataset
from mmengine.logging import print_log
from mmengine.registry import DATASETS


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
    """

    def __init__(
        self,
        metainfo,
        class_names,
        pointcloud_range: List[float],
        use_valid_flag: bool = False,
        repeat_sampling_factory_t: Optional[float] = None,
        **kwargs,
    ):
        T4Dataset.METAINFO = metainfo
        self.valid_class_name_ins = {class_name: 0 for class_name in class_names}

        # Number of frames for each category that contains at least one of the category
        self.category_frame_number = {class_name: 0 for class_name in class_names}
        self.class_names = class_names
        self.point_cloud_range = pointcloud_range
        super().__init__(use_valid_flag=use_valid_flag, **kwargs)
        print_log(f"Valid dataset instances: {self.valid_class_name_ins}", logger="current")

        # Total bbox
        total_bboxes = sum(self.valid_class_name_ins.values())

        # Compute bbox fraction
        self.valid_class_bbox_fraction = {
            class_name: class_bbox_num / total_bboxes
            for class_name, class_bbox_num in self.valid_class_name_ins.items()
        }
        print_log(f"Valid dataset fraction: {self.valid_class_bbox_fraction}", logger="current")
        self.repeat_sampling_factory_t = repeat_sampling_factory_t
        self.frame_weights = self._compute_frame_repeat_sampling_factor()
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

    def _compute_frame_repeat_sampling_factor(self) -> List[float]:
        """ """
        if self.repeat_sampling_factory_t is None:
            return [1.0] * len(self.data_list)

        # Compute category faction
        category_fraction_factor = {
            class_name: max(
                1,
                math.sqrt(
                    self.repeat_sampling_factory_t / (number_frame * self.valid_class_bbox_fraction[class_name])
                ),
            )
            for class_name, number_frame in self.category_frame_number
        }
        print_log(f"Category repeat weights: {category_fraction_factor}", logger="current")

        frame_weights = []
        for data in self.data_list:
            ann_info = data["ann_info"]
            valid_bbox_categories = ann_info["valid_bbox_categories"]
            frame_repeat_sampling_factor = 1
            for class_name, value in valid_bbox_categories.items():
                if value:
                    frame_repeat_sampling_factor = max(
                        frame_repeat_sampling_factor, category_fraction_factor[class_name]
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
        bbox_range = [
            self.point_cloud_range[0],
            self.point_cloud_range[1],
            self.point_cloud_range[3],
            self.point_cloud_range[4],
        ]
        bbox_in_ranges = ann_info["gt_bboxes_3d"].in_range_bev(bbox_range)
        valid_bbox_categories = {class_name: 0 for class_name in self.class_names}
        for label, bbox_in_ranges in zip(ann_info["gt_labels_3d"], bbox_in_ranges):
            if bbox_in_ranges:
                class_idx = self.class_names[label]
                self.valid_class_name_ins[class_idx] += 1
                # Set to 1 if a category exists in this frame
                valid_bbox_categories[class_idx] = 1

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

        return info
