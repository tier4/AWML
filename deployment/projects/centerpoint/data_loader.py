"""
CenterPoint DataLoader for deployment.

Wraps MMDet3D Dataset to ensure GT is identical to tools/detection3d/test.py.
Pipeline is run once per sample in load_sample(), avoiding redundant computation.
"""

import copy
from typing import Any, Dict, List, Optional, Union

import mmdet3d.datasets.transforms  # noqa: F401 - registers transforms
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, init_default_scope

from deployment.core import BaseDataLoader


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
        device: str = "cpu",
        task_type: Optional[str] = None,
    ):
        super().__init__(
            config={
                "info_file": info_file,
                "device": device,
            }
        )

        self.model_cfg = model_cfg
        self.device = device
        self.info_file = info_file
        self.dataset = self._build_dataset(model_cfg, info_file)

    def _build_dataset(self, model_cfg: Config, info_file: str) -> Any:
        """Build MMDet3D Dataset from config, overriding ann_file."""
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

    def _to_tensor(
        self,
        data: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
        name: str = "data",
    ) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(self.device)

        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)

        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError(f"Empty list for '{name}' in pipeline output.")

            first_item = data[0]
            if isinstance(first_item, torch.Tensor):
                return first_item.to(self.device)
            if isinstance(first_item, np.ndarray):
                return torch.from_numpy(first_item).to(self.device)

            raise ValueError(
                f"Unexpected type for {name}[0]: {type(first_item)}. Expected torch.Tensor or np.ndarray."
            )

        raise ValueError(
            f"Unexpected type for '{name}': {type(data)}. Expected torch.Tensor, np.ndarray, or list of tensors/arrays."
        )

    def load_sample(self, index: int) -> Dict[str, Any]:
        """Load sample by running the full pipeline once.

        Returns a dict containing all data needed for inference and evaluation:
        - points: Points tensor (ready for inference)
        - metainfo: Sample metadata
        - ground_truth: Raw eval_ann_info from MMDet3D (kept unconverted)
        """
        if index >= len(self.dataset):
            raise IndexError(f"Sample index {index} out of range (0-{len(self.dataset)-1})")

        # Run pipeline once
        data = self.dataset[index]

        # Extract inputs
        pipeline_inputs = data.get("inputs", {})
        if "points" not in pipeline_inputs:
            raise ValueError(f"Expected 'points' in inputs. Got keys: {list(pipeline_inputs.keys())}")

        points_tensor = self._to_tensor(pipeline_inputs["points"], name="points")
        if points_tensor.ndim != 2:
            raise ValueError(f"Expected points tensor with shape [N, features], got {points_tensor.shape}")

        # Extract metainfo and eval_ann_info from data_samples
        data_samples = data.get("data_samples")
        metainfo: Dict[str, Any] = {}
        ground_truth: Dict[str, Any] = {}

        if data_samples is not None:
            metainfo = getattr(data_samples, "metainfo", {}) or {}
            eval_ann_info = getattr(data_samples, "eval_ann_info", {}) or {}
            # Keep raw eval_ann_info here; evaluator will convert to the metrics format.
            ground_truth = dict(eval_ann_info)

        return {
            "points": points_tensor,
            "metainfo": metainfo,
            "ground_truth": ground_truth,
        }

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract points and metainfo from loaded sample.

        This is a lightweight operation - pipeline already ran in load_sample().
        """
        return {
            "points": sample["points"],
            "metainfo": sample["metainfo"],
        }

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    @property
    def class_names(self) -> List[str]:
        # Get from dataset's metainfo or model_cfg
        if hasattr(self.dataset, "metainfo") and "classes" in self.dataset.metainfo:
            return list(self.dataset.metainfo["classes"])

        if hasattr(self.model_cfg, "class_names"):
            return list(self.model_cfg.class_names)

        raise ValueError("class_names not found in dataset.metainfo or model_cfg")
