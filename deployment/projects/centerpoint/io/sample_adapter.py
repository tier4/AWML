from __future__ import annotations

import logging
from collections.abc import Mapping

import torch

from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.export_pipelines.interfaces import ExportSampleAdapter
from deployment.projects.centerpoint.io.sample_types import CenterPointFeatureSample, VoxelDict


class CenterPointSampleAdapter(ExportSampleAdapter):
    """Adapter for CenterPoint feature extraction output into typed sample payload."""

    _REQUIRED_VOXEL_KEYS: tuple[str, ...] = ("voxels", "num_points", "coors")

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize the sample adapter.

        Args:
            logger: Optional logger; defaults to module logger if not provided.
        """
        self.logger = logger or logging.getLogger(__name__)

    def extract_sample(
        self,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        sample_idx: int,
    ) -> CenterPointFeatureSample:
        """Extract a typed export sample from the model and data loader.

        Args:
            model: CenterPoint model with _extract_features (ONNX-compatible).
            data_loader: Loader used to fetch sample data.
            sample_idx: Index of the sample to extract.

        Returns:
            Typed CenterPointFeatureSample for export pipelines.

        Raises:
            AttributeError: If model does not have _extract_features.
            TypeError: If ``_extract_features`` return value has wrong types or shape.
            KeyError: If voxel_dict is missing required keys.
        """
        if not hasattr(model, "_extract_features"):
            raise AttributeError(
                "CenterPoint model must have _extract_features method for ONNX export. "
                "Please ensure the model is built with ONNX compatibility."
            )

        input_features, voxel_dict = model._extract_features(data_loader, sample_idx)

        if not isinstance(input_features, torch.Tensor):
            raise TypeError(f"input_features must be torch.Tensor, got {type(input_features).__name__}")
        if not isinstance(voxel_dict, Mapping):
            raise TypeError(f"voxel_dict must be Mapping, got {type(voxel_dict).__name__}")

        missing = [key for key in self._REQUIRED_VOXEL_KEYS if key not in voxel_dict]
        if missing:
            raise KeyError(f"voxel_dict missing keys: {missing}")

        invalid = {
            key: type(voxel_dict[key]).__name__
            for key in self._REQUIRED_VOXEL_KEYS
            if not isinstance(voxel_dict[key], torch.Tensor)
        }
        if invalid:
            raise TypeError(f"voxel_dict invalid tensor fields: {invalid}")

        validated_voxel_dict: VoxelDict = {k: voxel_dict[k] for k in self._REQUIRED_VOXEL_KEYS}

        return CenterPointFeatureSample(
            input_features=input_features,
            voxel_dict=validated_voxel_dict,
        )
