from __future__ import annotations

import logging
from typing import Any, cast

import torch

from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.export_pipelines.interfaces import ExportSampleAdapter
from deployment.projects.centerpoint.export_types import CenterPointExportSample, VoxelDict


class CenterPointSampleAdapter(ExportSampleAdapter):
    """Adapt legacy CenterPoint feature extraction output into typed sample payload."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def extract_sample(
        self,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        sample_idx: int,
    ) -> CenterPointExportSample:
        if not hasattr(model, "_extract_features"):
            raise AttributeError(
                "CenterPoint model must have _extract_features method for ONNX export. "
                "Please ensure the model is built with ONNX compatibility."
            )

        raw = model._extract_features(data_loader, sample_idx)
        return self._to_export_sample(raw)

    def _to_export_sample(self, raw: Any) -> CenterPointExportSample:
        if not (isinstance(raw, (tuple, list)) and len(raw) == 2):
            raise TypeError(
                "Invalid sample data for CenterPoint export. Expected "
                "(input_features: torch.Tensor, voxel_dict: dict)."
            )

        input_features, voxel_dict = raw
        if not isinstance(input_features, torch.Tensor):
            raise TypeError(f"input_features must be torch.Tensor, got: {type(input_features)}")
        if not isinstance(voxel_dict, dict):
            raise TypeError(f"voxel_dict must be dict, got: {type(voxel_dict)}")

        for key in ("voxels", "num_points", "coors"):
            if key not in voxel_dict:
                raise KeyError(f"voxel_dict must contain key '{key}'")
            if not isinstance(voxel_dict[key], torch.Tensor):
                raise TypeError(f"voxel_dict['{key}'] must be torch.Tensor, got: {type(voxel_dict[key])}")

        return CenterPointExportSample(
            input_features=input_features,
            voxel_dict=cast(VoxelDict, voxel_dict),
        )
