from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TypedDict

import torch

from deployment.core.io.base_data_loader import SampleData


class CenterPointSample(SampleData):
    """Structured payload after running the MMDet3D test pipeline for one frame.

    Returned by :meth:`deployment.projects.centerpoint.io.data_loader.CenterPointDataLoader.load_sample`.
    At runtime this is a plain ``dict``; use bracket access (e.g. ``sample["points"]``).

    Attributes:
        points: Point cloud tensor on CPU, shape ``[N, C]`` after pipeline.
        metainfo: Per-sample metadata (e.g. lidar path, sample index) as a string-keyed dict.
        ground_truth: Raw ``eval_ann_info`` from the detector data sample, for evaluation.
    """

    points: torch.Tensor
    metainfo: Dict[str, object]
    ground_truth: Dict[str, object]


class CenterPointModelInput(TypedDict):
    """Subset of a loaded sample passed into the CenterPoint network for inference.

    Produced by :meth:`deployment.projects.centerpoint.io.data_loader.CenterPointDataLoader.preprocess`.
    Excludes ``ground_truth``, which is only needed for eval/export wiring.

    Attributes:
        points: Point cloud tensor for the model forward.
        metainfo: Metadata required by preprocessing or postprocessing.
    """

    points: torch.Tensor
    metainfo: Dict[str, object]


class VoxelDict(TypedDict):
    """Voxelization output from CenterPoint feature extraction (ONNX/export path).

    Matches the dict returned alongside ``input_features`` from ``_extract_features``.

    Attributes:
        voxels: Packed voxel feature tensor.
        num_points: Per-voxel point counts.
        coors: Voxel coordinates (e.g. batch and grid indices).
    """

    voxels: torch.Tensor
    num_points: torch.Tensor
    coors: torch.Tensor


@dataclass(frozen=True)
class CenterPointFeatureSample:
    """Immutable bundle of backbone inputs and sparse tensor layout for export.

    Built by `deployment.projects.centerpoint.io.sample_adapter.CenterPointSampleAdapter`
    for ONNX/TensorRT pipelines that need validated tensors and a consistent voxel dict.

    Attributes:
        input_features: Tensor fed to the rest of the network after voxelization.
        voxel_dict: Sparse structure with keys ``voxels``, ``num_points``, ``coors``.
    """

    input_features: torch.Tensor
    voxel_dict: VoxelDict

    @property
    def coors(self) -> torch.Tensor:
        return self.voxel_dict["coors"]
