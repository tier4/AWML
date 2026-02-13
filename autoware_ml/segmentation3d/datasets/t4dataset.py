from os import path as osp
from typing import Callable, List, Optional, Union

import numpy as np
from mmdet3d.registry import DATASETS
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_local_path

from autoware_ml.segmentation3d.datasets.utils import class_mapping_to_names_palette_label2cat


@DATASETS.register_module()
class T4SegDataset(BaseDataset):
    """T4 Dataset for 3D semantic segmentation.

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='points',
                 img='',
                 pts_instance_mask='',
                 pts_semantic_mask='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used
            as input, it usually has following keys:

                - use_camera: bool
                - use_lidar: bool
            Defaults to dict(use_lidar=True, use_camera=False).
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.classes) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (np.ndarray or str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        serialize_data (bool): Whether to hold memory using serialized objects,
            when enabled, data loader workers can use shared RAM from master
            process instead of making a copy.
            Defaults to False for 3D Segmentation datasets.
        load_eval_anns (bool): Whether to load annotations in test_mode,
            the annotation will be save in `eval_ann_infos`, which can be used
            in Evaluator. Defaults to True.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        lidar_sources (List[str], optional): List of LiDAR source names to use.
            When specified, each frame is expanded into N separate samples
            (one per source). The transforms will receive which source to load
            via ``lidar_sources_to_load`` in data_info.
            When None, all points are loaded as-is without source filtering.
            Defaults to None.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        ann_file: str = "",
        metainfo: Optional[dict] = None,
        data_prefix: dict = dict(pts="", img="", pts_instance_mask="", pts_semantic_mask=""),
        pipeline: List[Union[dict, Callable]] = [],
        modality: dict = dict(use_lidar=True, use_camera=False),
        ignore_index: Optional[int] = None,
        scene_idxs: Optional[Union[str, np.ndarray]] = None,
        test_mode: bool = False,
        serialize_data: bool = False,
        load_eval_anns: bool = True,
        backend_args: Optional[dict] = None,
        lidar_sources: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.backend_args = backend_args
        self.modality = modality
        self.load_eval_anns = load_eval_anns
        self.ignore_index = ignore_index
        self.lidar_sources = lidar_sources
        self._num_sources = len(lidar_sources) if lidar_sources else 1

        class_mapping = metainfo.get("class_mapping", None)
        base_class_names = metainfo.get("base_class_names", None)
        base_palette = metainfo.get("base_palette", None)
        metainfo["ignore_index"] = self.ignore_index
        class_names, palette, label2cat = class_mapping_to_names_palette_label2cat(
            class_mapping=class_mapping,
            ignore_index=self.ignore_index,
            base_palette=base_palette,
            base_class_names=base_class_names,
        )
        metainfo["class_names"] = class_names
        metainfo["palette"] = palette
        metainfo["label2cat"] = label2cat

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=serialize_data,
            **kwargs,
        )

        if not kwargs.get("lazy_init", False):
            self.scene_idxs = self.get_scene_idxs(scene_idxs)
            self.data_list = [self.data_list[i] for i in self.scene_idxs]

            # set group flag for the sampler
            if not self.test_mode:
                self._set_group_flag()

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process
        the `instances` field to `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality["use_lidar"]:
            info["lidar_points"]["lidar_path"] = osp.join(
                self.data_prefix.get("pts", ""), info["lidar_points"]["lidar_path"]
            )
            if "num_pts_feats" in info["lidar_points"]:
                info["num_pts_feats"] = info["lidar_points"]["num_pts_feats"]
            info["lidar_path"] = info["lidar_points"]["lidar_path"]

            if "pts_semantic_mask_path" in info:
                info["pts_semantic_mask_path"] = osp.join(
                    self.data_prefix.get("pts_semantic_mask", ""), info["pts_semantic_mask_path"]
                )

        if self.modality["use_camera"]:
            for cam_id, img_info in info["images"].items():
                if "img_path" in img_info:
                    img_info["img_path"] = osp.join(self.data_prefix.get("img", ""), img_info["img_path"])

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode and self.load_eval_anns:
            info["eval_ann_info"] = dict()

        return info

    def _map_index(self, idx: int) -> tuple:
        """Map expanded index to (frame_idx, source_idx).

        Args:
            idx (int): The expanded index.

        Returns:
            tuple: (frame_idx, source_idx) where source_idx is None if not expanded.
        """
        if self.lidar_sources is not None:
            frame_idx = idx // self._num_sources
            source_idx = idx % self._num_sources
            return frame_idx, source_idx
        return idx, None

    def get_data_info(self, idx: int) -> dict:
        """Get data info by index.

        When ``lidar_sources`` is set, the index is mapped from the expanded
        dataset space to a frame index, and the selected source is injected
        into data_info as ``lidar_sources_to_load``.

        Args:
            idx (int): The index of ``data_info``. When lidar_sources is set,
                this is an expanded index that maps to a specific frame and source.

        Returns:
            dict: Data info dict.
        """
        frame_idx, source_idx = self._map_index(idx)
        # Call parent's get_data_info with frame index
        data_info = super().get_data_info(frame_idx)

        # Inject source information
        if source_idx is not None:
            data_info["lidar_sources_to_load"] = [self.lidar_sources[source_idx]]
        else:
            data_info["lidar_sources_to_load"] = None

        return data_info

    def prepare_data(self, idx: int) -> dict:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``. When lidar_sources is set,
                this is an expanded index that maps to a specific frame and source.

        Returns:
            dict: Results passed through ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        data_info["dataset"] = self
        return self.pipeline(data_info)

    def __len__(self) -> int:
        """Return the expanded dataset length.

        When ``lidar_sources`` is set, the length is multiplied by the number
        of sources, as each frame becomes N separate samples.

        Returns:
            int: The total number of samples in the dataset.
        """
        base_len = len(self.data_list)
        if self.lidar_sources is not None:
            return base_len * self._num_sources
        return base_len

    def get_scene_idxs(self, scene_idxs: Union[None, str, np.ndarray]) -> np.ndarray:
        """Compute scene_idxs for data sampling.

        We sample more times for scenes with more points.
        Note: scene_idxs operates on frame indices (base data_list), not expanded indices.
        """
        # Use base frame count, not expanded length
        base_len = len(self.data_list)

        if self.test_mode:
            # when testing, we load one whole scene every time
            return np.arange(base_len).astype(np.int32)

        # we may need to re-sample different scenes according to scene_idxs
        # this is necessary for indoor scene segmentation such as ScanNet
        if scene_idxs is None:
            scene_idxs = np.arange(base_len)
        if isinstance(scene_idxs, str):
            scene_idxs = osp.join(self.data_root, scene_idxs)
            with get_local_path(scene_idxs, backend_args=self.backend_args) as local_path:
                scene_idxs = np.load(local_path)
        else:
            scene_idxs = np.array(scene_idxs)

        return scene_idxs.astype(np.int32)

    def _set_group_flag(self) -> None:
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        # Use explicit length calculation to avoid calling __len__ before full init
        base_len = len(self.data_list)
        total_len = base_len * self._num_sources if self.lidar_sources else base_len
        self.flag = np.zeros(total_len, dtype=np.uint8)
