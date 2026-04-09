"""
CenterPoint DataLoader for deployment.

Wraps MMDet3D Dataset to ensure GT is identical to tools/detection3d/test.py.
Pipeline is run once per sample in load_sample(), avoiding redundant computation.
"""

import copy
from typing import Any, List, Mapping

import mmdet3d.datasets.transforms  # noqa: F401 - registers transforms
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, init_default_scope
from typing_extensions import override

from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.projects.centerpoint.io.sample_types import (
    CenterPointModelInput,
    CenterPointSample,
)


def _require_key(mapping: Mapping[str, Any], key: str, owner: str) -> Any:
    try:
        return mapping[key]
    except KeyError as e:
        raise KeyError(f"{owner} must contain '{key}'. Got keys: {list(mapping.keys())}") from e


def _require_attr(obj: Any, attr: str, owner: str) -> Any:
    if not hasattr(obj, attr):
        raise AttributeError(f"{owner}.{attr} is required.")
    value = getattr(obj, attr)
    if value is None:
        raise ValueError(f"{owner}.{attr} must not be None.")
    return value


class CenterPointDataLoader(BaseDataLoader):
    """Deployment dataloader for CenterPoint using MMDet3D Dataset.

    This wraps the same Dataset used by tools/detection3d/test.py, ensuring:
    - GT is identical
    - Pipeline processing is identical
    - Pipeline runs once per sample (no cache needed)

    Design:
        load_sample() runs the full pipeline and returns all data (input + GT).
        preprocess() extracts model inputs from the loaded sample.
    """

    def __init__(
        self,
        info_file: str,
        model_cfg: Config,
    ) -> None:
        """Initialize CenterPoint data loader.

        Args:
            info_file: Path to dataset info file (e.g. pkl) used as ann_file.
            model_cfg: MMEngine model config; must have test_dataloader.dataset.
        """
        super().__init__()

        self.model_cfg = model_cfg
        self.device = "cpu"
        self.info_file = info_file
        self.dataset = self._build_dataset(model_cfg, info_file)

    def _build_dataset(self, model_cfg: Config, info_file: str) -> torch.utils.data.Dataset:
        """Build MMDet3D Dataset from config, overriding ann_file.

        Args:
            model_cfg: MMEngine model config with test_dataloader.dataset.
            info_file: Path to dataset info file (e.g. pkl) used as ann_file.

        Returns:
            Built MMDet3D Dataset instance.

        Raises:
            ValueError: If model_cfg does not have test_dataloader.
        """
        # Set default scope to mmdet3d so transforms are found in the registry
        init_default_scope("mmdet3d")
        if not hasattr(model_cfg, "test_dataloader"):
            raise ValueError("model_cfg must have 'test_dataloader' with dataset config")

        dataset_cfg = copy.deepcopy(model_cfg.test_dataloader.dataset)

        dataset_cfg["ann_file"] = info_file
        dataset_cfg["test_mode"] = True

        # Build dataset
        dataset = DATASETS.build(dataset_cfg)
        return dataset

    @override
    def load_sample(self, index: int) -> CenterPointSample:
        """Load sample by running the full pipeline once.

        Returns a dict containing all data needed for inference and evaluation:
        - points: Points tensor (ready for inference)
        - metainfo: Sample metadata
        - ground_truth: Raw eval_ann_info from MMDet3D (kept unconverted)

        Args:
            index: Sample index in the dataset (0 to num_samples - 1).

        Returns:
            :class:`CenterPointSample` with keys ``points``, ``metainfo``, ``ground_truth``.

        Raises:
            IndexError: If index is out of range.
            KeyError: If dataset sample is missing required keys.
            ValueError: If ``data_samples`` is None, required attributes are None, or points shape is invalid.
            AttributeError: If ``data_samples`` lacks required attributes.
        """
        if index >= len(self.dataset):
            raise IndexError(f"Sample index {index} out of range (0-{len(self.dataset)-1})")

        # Run pipeline once
        data = self.dataset[index]

        pipeline_inputs = _require_key(data, "inputs", "Dataset sample")
        points_tensor = _require_key(pipeline_inputs, "points", "inputs").to("cpu")
        if points_tensor.ndim != 2:
            raise ValueError(f"Expected points tensor with shape [N, features], got {points_tensor.shape}")

        data_samples = _require_key(data, "data_samples", "Dataset sample")
        if data_samples is None:
            raise ValueError("Dataset sample contains None 'data_samples', cannot build evaluation ground truth.")

        metainfo = _require_attr(data_samples, "metainfo", "data_samples")
        eval_ann_info = _require_attr(data_samples, "eval_ann_info", "data_samples")
        # Keep raw eval_ann_info here; evaluator will convert to the metrics format.
        ground_truth = dict(eval_ann_info)

        return CenterPointSample(
            points=points_tensor,
            metainfo=dict(metainfo),
            ground_truth=ground_truth,
        )

    @override
    def preprocess(self, sample: CenterPointSample) -> CenterPointModelInput:
        """Extract points and metainfo from loaded sample.

        This is a lightweight operation - pipeline already ran in load_sample().

        Args:
            sample: Result of :meth:`load_sample` with keys ``points`` and ``metainfo``.

        Returns:
            Dict with keys ``points`` and ``metainfo`` for inference.
        """
        return CenterPointModelInput(
            points=sample["points"],
            metainfo=sample["metainfo"],
        )

    @property
    @override
    def num_samples(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    @property
    def class_names(self) -> List[str]:
        """Return class names from dataset metainfo or model_cfg.

        Returns:
            List of class name strings.

        Raises:
            ValueError: If class_names not found in dataset.metainfo or model_cfg.
        """
        # Get from dataset's metainfo or model_cfg
        if hasattr(self.dataset, "metainfo") and "classes" in self.dataset.metainfo:
            return list(self.dataset.metainfo["classes"])

        if hasattr(self.model_cfg, "class_names"):
            return list(self.model_cfg.class_names)

        raise ValueError("class_names not found in dataset.metainfo or model_cfg")
