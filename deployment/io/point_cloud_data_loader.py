"""Shared MMDet3D point-cloud data loader for deployment.

Point-cloud detectors (CenterPoint, BEVFusion, …) all wrap the same MMDet3D test dataset:
build it from ``model_cfg.test_dataloader.dataset``, run the pipeline once per sample, and
return ``points`` / ``metainfo`` / ``ground_truth``. Because that flow is identical, both
projects use this loader directly instead of subclassing it.

TODO(vividf): BEVFusion is multi-modal (lidar + camera) but is deployed lidar-only today, so
this point-cloud loader is sufficient. Once camera inputs are added, give BEVFusion its own
loader (or extend this one) to load images and the lidar2img transforms alongside ``points``.
"""

from __future__ import annotations

import copy
from typing import Type

import mmdet3d.datasets.transforms  # noqa: F401 - registers transforms in the mmdet3d registry
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, init_default_scope
from typing_extensions import override

from deployment.io.base_data_loader import BaseDataLoader, SampleData


class PointCloudDataLoader(BaseDataLoader):
    """DataLoader that runs the MMDet3D test pipeline once per sample.

    The runtime payload is always ``points`` / ``metainfo`` (+ ``ground_truth`` for the loaded
    sample); :attr:`sample_cls` / :attr:`model_input_cls` are the typed dicts describing it and
    can be overridden if a project needs a richer typed payload.
    """

    #: Typed payloads returned by ``load_sample`` / ``preprocess`` (override in subclasses).
    sample_cls: Type[SampleData] = SampleData
    model_input_cls: Type[SampleData] = SampleData

    def __init__(self, model_cfg: Config, info_file: str = "") -> None:
        """Build the MMDet3D dataset used for deployment evaluation.

        Args:
            model_cfg: MMEngine model config; must have ``test_dataloader.dataset``.
            info_file: Optional eval info file overriding the dataset's ``ann_file``; empty
                keeps the model config's own ``ann_file``.
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.info_file = info_file
        self.dataset = self._build_dataset(model_cfg, info_file)

    def _build_dataset(self, model_cfg: Config, info_file: str) -> torch.utils.data.Dataset:
        init_default_scope("mmdet3d")
        if not hasattr(model_cfg, "test_dataloader"):
            raise ValueError("model_cfg must have 'test_dataloader' with dataset config")
        dataset_cfg = copy.deepcopy(model_cfg.test_dataloader.dataset)
        # Only override the eval info file when a deploy config supplies one; otherwise fall back
        # to the model config's own ``ann_file``.
        if info_file:
            dataset_cfg["ann_file"] = info_file
        dataset_cfg["test_mode"] = True
        return DATASETS.build(dataset_cfg)

    @override
    def load_sample(self, index: int) -> SampleData:
        if index >= len(self.dataset):
            raise IndexError(f"Sample index {index} out of range (0-{len(self.dataset)-1})")

        data = self.dataset[index]
        points_tensor = data["inputs"]["points"].to("cpu")
        if points_tensor.ndim != 2:
            raise ValueError(f"Expected points tensor with shape [N, features], got {points_tensor.shape}")

        data_samples = data["data_samples"]
        if data_samples is None:
            raise ValueError("Dataset sample contains None 'data_samples', cannot build evaluation ground truth.")

        metainfo = getattr(data_samples, "metainfo", None)
        eval_ann_info = getattr(data_samples, "eval_ann_info", None)
        return self.sample_cls(
            points=points_tensor,
            metainfo=dict(metainfo) if metainfo else {},
            ground_truth=dict(eval_ann_info) if eval_ann_info else {},
        )

    @override
    def preprocess(self, sample: SampleData) -> SampleData:
        return self.model_input_cls(points=sample["points"], metainfo=sample["metainfo"])

    @property
    @override
    def num_samples(self) -> int:
        return len(self.dataset)
