from os import path as osp

from mmdet3d.datasets import NuScenesDataset
from mmengine.registry import DATASETS


@DATASETS.register_module()
class T4Dataset(NuScenesDataset):
    """T4Dataset Dataset base class
    The descriptions below are for the methods that aren't implemented this class.
    """

    def __init__(
        self,
        metainfo,
        class_names,
        **kwargs,
    ):
        T4Dataset.METAINFO = metainfo
        super().__init__(**kwargs)
        self.class_names = class_names

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
        self.modality["use_lidar"] = False
        info = super().parse_data_info(info)
        self.modality["use_lidar"] = use_lidar

        # modified from https://github.com/open-mmlab/mmdetection3d/blob/v1.2.0/mmdet3d/datasets/det3d_dataset.py#L279-L296
        if self.modality["use_lidar"]:
            info["lidar_points"]["lidar_path"] = osp.join(
                self.data_prefix.get("pts", ""),
                info["lidar_points"]["lidar_path"])
            info["num_pts_feats"] = info["lidar_points"]["num_pts_feats"]
            info["lidar_path"] = info["lidar_points"]["lidar_path"]
            if "lidar_sweeps" in info:
                for sweep in info["lidar_sweeps"]:
                    # NOTE: modified to avoid hard-coded processes for nuscenes dataset
                    file_suffix = sweep["lidar_points"]["lidar_path"]
                    # -----------------------------------------------
                    if "samples" in sweep["lidar_points"]["lidar_path"]:
                        sweep["lidar_points"]["lidar_path"] = osp.join(
                            self.data_prefix["pts"], file_suffix)
                    else:
                        sweep["lidar_points"]["lidar_path"] = osp.join(
                            self.data_prefix["sweeps"], file_suffix)
        return info
