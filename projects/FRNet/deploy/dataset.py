"""Dataset loaders for FRNet deployment.

Unified API for loading point clouds, ground-truth labels and class metadata
from NuScenes and T4Dataset formats.  All loaders share the same interface
so the rest of the pipeline doesn't need to know which dataset is used.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import mmengine
import numpy as np
import numpy.typing as npt
from mmengine.config import Config
from mmengine.logging import MMLogger
from nuscenes.nuscenes import NuScenes

from autoware_ml.segmentation3d.datasets.transforms.loading import LoadPointsWithIdentifierFromFile
from autoware_ml.segmentation3d.datasets.utils import (
    class_mapping_to_names_palette_label2cat,
    load_and_map_semantic_mask,
)


class DatasetLoader:
    """Base class for dataset-specific point cloud loading and metadata.

    Subclasses must override ``_build_class_info`` to provide class metadata
    and implement ``get_samples``, ``load_points``, and ``load_gt`` for
    dataset-specific I/O.
    """

    def __init__(self, model_cfg: Config, dataset_dir: str) -> None:
        self._model_cfg = model_cfg
        self._dataset_dir = dataset_dir
        self.logger = MMLogger.get_current_instance()
        self._class_names, self._palette, self._label2cat = self._build_class_info()

    @property
    def class_names(self) -> List[str]:
        """Ordered class names including the unknown class at ignore_index."""
        return self._class_names

    @property
    def palette(self) -> List[List[int]]:
        """RGB palette aligned with class_names (same length)."""
        return self._palette

    @property
    def label2cat(self) -> Dict[int, str]:
        """Mapping from label index to class name, used by seg_eval."""
        return self._label2cat

    def _build_class_info(self) -> tuple:
        """Return (class_names, palette, label2cat).  Must be overridden."""
        raise NotImplementedError

    def get_samples(self) -> List[Dict[str, Any]]:
        """Return list of sample dicts for load_points/load_gt."""
        raise NotImplementedError

    def load_points(self, sample: Dict[str, Any]) -> npt.NDArray[np.float32]:
        """Load point cloud for the given sample, shape (N, D)."""
        raise NotImplementedError

    def load_gt(self, sample: Dict[str, Any]) -> Optional[npt.NDArray[np.int64]]:
        """Load ground-truth semantic labels, shape (N,). Returns None if unavailable."""
        raise NotImplementedError


class NuScenesLoader(DatasetLoader):
    """NuScenes lidar-segmentation dataset loader.

    Loads point clouds from .bin files using the nuscenes-devkit.
    GT labels come from the lidarseg table (available in v1.0-mini and
    v1.0-trainval; v1.0-test has no GT so load_gt returns None).
    """

    _PALETTE: List[List[int]] = [
        [255, 120, 50],  # barrier
        [255, 192, 203],  # bicycle
        [255, 255, 0],  # bus
        [0, 150, 245],  # car
        [0, 255, 255],  # construction_vehicle
        [255, 127, 0],  # motorcycle
        [255, 0, 0],  # pedestrian
        [255, 240, 150],  # traffic_cone
        [135, 60, 0],  # trailer
        [160, 32, 240],  # truck
        [255, 0, 255],  # driveable_surface
        [139, 137, 137],  # other_flat
        [75, 0, 75],  # sidewalk
        [150, 240, 80],  # terrain
        [230, 230, 250],  # manmade
        [0, 175, 0],  # vegetation
        [0, 0, 0],  # unknown
    ]

    def __init__(self, model_cfg: Config, dataset_dir: str) -> None:
        super().__init__(model_cfg, dataset_dir)
        self._labels_map = self._build_labels_map()

    def _build_class_info(self) -> tuple:
        """Build class names, palette and label2cat for NuScenes."""
        class_names = list(self._model_cfg.class_names) + ["unknown"]
        label2cat = {i: n for i, n in enumerate(self._model_cfg.class_names)}
        return class_names, self._PALETTE, label2cat

    def get_samples(self) -> List[Dict[str, Any]]:
        """Load scenes from NuScenes and return one sample per scene."""
        nusc = NuScenes(version="v1.0-test", dataroot=self._dataset_dir, verbose=True)

        has_lidarseg = "lidarseg" in nusc.table_names and len(nusc.lidarseg) > 0
        if has_lidarseg:
            self.logger.info("Lidarseg annotations available – GT evaluation enabled.")
        else:
            self.logger.warning("Lidarseg annotations NOT available – GT evaluation disabled.")

        samples: List[Dict[str, Any]] = []
        self.logger.info(f"Total test scenes: {len(nusc.scene)}")
        for scene in nusc.scene:
            scene_rec = nusc.get("scene", scene["token"])
            sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
            sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
            lidar_path, _, _ = nusc.get_sample_data(sd_rec["token"])

            sample_dict: Dict[str, Any] = {"pcd_path": lidar_path}

            # Attach lidarseg path when GT is available
            if has_lidarseg:
                try:
                    lidarseg_rec = nusc.get("lidarseg", sd_rec["token"])
                    sample_dict["lidarseg_path"] = os.path.join(self._dataset_dir, lidarseg_rec["filename"])
                except KeyError:
                    pass  # This particular sample has no GT

            samples.append(sample_dict)

        self.logger.info(f"Total samples: {len(samples)}")
        return samples

    def load_points(self, sample: Dict[str, Any]) -> npt.NDArray[np.float32]:
        """Read .bin file and return (N, 4) points [x, y, z, intensity]."""
        points = np.fromfile(sample["pcd_path"], dtype=np.float32).reshape(-1, 5)
        return points[:, :4]

    def load_gt(self, sample: Dict[str, Any]) -> Optional[npt.NDArray[np.int64]]:
        """Load lidarseg GT labels and remap via seg_label_mapping LUT."""
        lidarseg_path = sample.get("lidarseg_path")
        if lidarseg_path is None or not os.path.isfile(lidarseg_path):
            return None

        raw_labels = np.fromfile(lidarseg_path, dtype=np.uint8)
        return self._apply_labels_map(raw_labels)

    def _build_labels_map(self) -> np.ndarray:
        """Build a numpy LUT from metainfo.seg_label_mapping."""
        mapping: Dict[int, int] = dict(self._model_cfg.metainfo.seg_label_mapping)
        max_raw = max(mapping.keys())
        lut = np.full(max_raw + 1, self._model_cfg.ignore_index, dtype=np.int64)
        for raw_label, target in mapping.items():
            lut[raw_label] = target
        return lut

    def _apply_labels_map(self, raw_labels: npt.NDArray[np.uint8]) -> npt.NDArray[np.int64]:
        """Map raw NuScenes labels to class indices via the LUT."""
        return self._labels_map[raw_labels]


class T4DatasetLoader(DatasetLoader):
    """T4 lidar-segmentation dataset loader.

    T4 stores merged multi-LiDAR point clouds in one file.  This handler
    splits them back per configured source using LoadPointsWithIdentifierFromFile
    and transforms each cloud into the sensor frame.

    GT labels are loaded via load_and_map_semantic_mask which slices the
    merged annotation to the source selection and remaps categories.
    """

    def __init__(self, model_cfg: Config, dataset_dir: str) -> None:
        super().__init__(model_cfg, dataset_dir)
        self._load_transform = self._build_load_transform()

    def get_samples(self) -> List[Dict[str, Any]]:
        """Load annotation pickle and expand each frame by lidar_sources."""
        ann_file = self._model_cfg.val_dataloader.dataset.ann_file
        ann_path = os.path.join(self._dataset_dir, ann_file)
        lidar_sources = self._model_cfg.val_dataloader.dataset.get("lidar_sources", None)
        if lidar_sources is None:
            raise ValueError(
                "T4SegDataset deployment requires 'lidar_sources' in the dataset config. "
                "Please set SOURCES in the model config."
            )

        self.logger.info(f"Loading annotations from: {ann_path}")
        self.logger.info(f"Configured LiDAR sources: {lidar_sources}")
        data_list = mmengine.load(ann_path)["data_list"]

        samples: List[Dict[str, Any]] = []
        for info in data_list:
            pcd_path = os.path.join(self._dataset_dir, info["lidar_points"]["lidar_path"])
            mask_path = os.path.join(self._dataset_dir, info.get("pts_semantic_mask_path"))
            mask_categories = info.get("pts_semantic_mask_categories")
            for channel in lidar_sources:
                samples.append(
                    {
                        "pcd_path": pcd_path,
                        "source_name": channel,
                        "lidar_sources": info.get("lidar_sources", {}),
                        "lidar_sources_info": info.get("lidar_sources_info", {}),
                        "pts_semantic_mask_path": mask_path,
                        "pts_semantic_mask_categories": mask_categories,
                    }
                )

        self.logger.info(f"Total frames: {len(data_list)}, samples (frames x sources): {len(samples)}")
        return samples

    def load_points(self, sample: Dict[str, Any]) -> npt.NDArray[np.float32]:
        """Load merged .bin, extract one source and transform to sensor frame."""
        source_name = sample["source_name"]
        results = {
            "lidar_points": {"lidar_path": sample["pcd_path"]},
            "lidar_sources_to_load": [source_name],
            "lidar_sources": sample.get("lidar_sources", {}),
            "lidar_sources_info": sample.get("lidar_sources_info", {}),
        }
        results = self._load_transform.transform(results)
        return results["points"].numpy()

    def load_gt(self, sample: Dict[str, Any]) -> Optional[npt.NDArray[np.int64]]:
        """Load merged GT mask, slice to source and remap via class_mapping."""
        mask_path = os.path.join(self._dataset_dir, sample.get("pts_semantic_mask_path"))
        mask_categories = sample.get("pts_semantic_mask_categories")
        if mask_path is None or mask_categories is None:
            return None

        # The pickle stores paths relative to the workspace root (e.g.
        # "data/t4dataset/…"), not relative to dataset_dir.  Use the path
        # as-is if it already exists, otherwise try joining with dataset_dir.
        if not os.path.isfile(mask_path):
            mask_path = os.path.join(self._dataset_dir, mask_path)
        if not os.path.isfile(mask_path):
            return None

        selection = self._get_source_selection(sample)
        selections = [selection] if selection is not None else None

        return load_and_map_semantic_mask(
            mask_path=mask_path,
            raw_categories=mask_categories,
            class_mapping=self._model_cfg.class_mapping,
            ignore_index=self._model_cfg.ignore_index,
            seg_dtype=np.uint8,
            selections=selections,
        )

    def _build_load_transform(self):
        """Build LoadPointsWithIdentifierFromFile from the test pipeline config."""
        load_cfg = next(c for c in self._model_cfg.test_pipeline if c["type"] == "LoadPointsWithIdentifierFromFile")
        cfg = dict(load_cfg)
        cfg.pop("type")
        return LoadPointsWithIdentifierFromFile(**cfg)

    def _build_class_info(self) -> tuple:
        """Derive class names, palette and label2cat from class_mapping config."""
        names, palette, label2cat = class_mapping_to_names_palette_label2cat(
            class_mapping=self._model_cfg.class_mapping,
            ignore_index=self._model_cfg.ignore_index,
            base_palette=self._model_cfg.metainfo.base_palette,
            base_class_names=self._model_cfg.metainfo.base_class_names,
        )
        names.append("unknown")
        palette.append([0, 0, 0])
        return names, palette, label2cat

    @staticmethod
    def _get_source_selection(sample: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Get {idx_begin, length} for the sample's source, mirroring the load transform."""
        source_name = sample.get("source_name")
        if source_name is None:
            return None

        source_map = sample.get("lidar_sources", {})
        sources_info = sample.get("lidar_sources_info", {})
        sources = sources_info.get("sources", [])
        if not sources or source_name not in source_map:
            return None

        sensor_token = source_map[source_name]["sensor_token"]
        token_to_range = {s["sensor_token"]: (s["idx_begin"], s["length"]) for s in sources}
        if sensor_token not in token_to_range:
            return None

        idx_begin, length = token_to_range[sensor_token]
        return {"idx_begin": idx_begin, "length": length}


_LOADERS: Dict[str, type] = {
    "NuScenesSegDataset": NuScenesLoader,
    "T4SegDataset": T4DatasetLoader,
}


def create_dataset_loader(model_cfg: Config, dataset_dir: str) -> DatasetLoader:
    """Create the right DatasetLoader based on model_cfg.dataset_type."""
    dataset_type = model_cfg.dataset_type
    if dataset_type not in _LOADERS:
        raise ValueError(f"Unsupported dataset_type: {dataset_type!r}. Supported: {list(_LOADERS)}")
    return _LOADERS[dataset_type](model_cfg, dataset_dir)
