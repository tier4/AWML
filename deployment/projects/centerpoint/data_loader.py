"""
CenterPoint DataLoader for deployment.

Moved from projects/CenterPoint/deploy/data_loader.py into the unified deployment bundle.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from mmengine.config import Config

from deployment.core import BaseDataLoader, build_preprocessing_pipeline


class CenterPointDataLoader(BaseDataLoader):
    """Deployment dataloader for CenterPoint.

    Responsibilities:
    - Load `info_file` (pickle) entries describing samples.
    - Build and run the MMEngine preprocessing pipeline for each sample.
    - Provide `load_sample()` for export helpers that need raw sample metadata.
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

        if not os.path.exists(info_file):
            raise FileNotFoundError(f"Info file not found: {info_file}")

        self.info_file = info_file
        self.model_cfg = model_cfg
        self.device = device

        self.data_infos = self._load_info_file()
        self.pipeline = build_preprocessing_pipeline(model_cfg, task_type=task_type)

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

    def _load_info_file(self) -> list:
        try:
            with open(self.info_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load info file: {e}") from e

        if isinstance(data, dict):
            if "data_list" in data:
                data_list = data["data_list"]
            elif "infos" in data:
                data_list = data["infos"]
            else:
                raise ValueError(f"Expected 'data_list' or 'infos' in info file, found keys: {list(data.keys())}")
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError(f"Unexpected info file format: {type(data)}")

        if not data_list:
            raise ValueError("No samples found in info file")

        return data_list

    def load_sample(self, index: int) -> Dict[str, Any]:
        if index >= len(self.data_infos):
            raise IndexError(f"Sample index {index} out of range (0-{len(self.data_infos)-1})")

        info = self.data_infos[index]

        lidar_points = info.get("lidar_points", {})
        if not lidar_points:
            lidar_path = info.get("lidar_path", info.get("velodyne_path", ""))
            lidar_points = {"lidar_path": lidar_path}

        if "lidar_path" in lidar_points and not lidar_points["lidar_path"].startswith("/"):
            data_root = getattr(self.model_cfg, "data_root", "data/t4dataset/")
            if not data_root.endswith("/"):
                data_root += "/"
            if not lidar_points["lidar_path"].startswith(data_root):
                lidar_points["lidar_path"] = data_root + lidar_points["lidar_path"]

        instances = info.get("instances", [])

        sample = {
            "lidar_points": lidar_points,
            "sample_idx": info.get("sample_idx", index),
            "timestamp": info.get("timestamp", 0),
        }

        if instances:
            gt_bboxes_3d = []
            gt_labels_3d = []

            for instance in instances:
                if "bbox_3d" in instance and "bbox_label_3d" in instance:
                    if instance.get("bbox_3d_isvalid", True):
                        gt_bboxes_3d.append(instance["bbox_3d"])
                        gt_labels_3d.append(instance["bbox_label_3d"])

            if gt_bboxes_3d:
                sample["gt_bboxes_3d"] = np.array(gt_bboxes_3d, dtype=np.float32)
                sample["gt_labels_3d"] = np.array(gt_labels_3d, dtype=np.int64)

        if "images" in info or "img_path" in info:
            sample["images"] = info.get("images", {})
            if "img_path" in info:
                sample["img_path"] = info["img_path"]

        return sample

    def preprocess(self, sample: Dict[str, Any]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        results = self.pipeline(sample)

        if "inputs" not in results:
            raise ValueError(
                "Expected 'inputs' key in pipeline results (MMDet3D 3.x format). "
                f"Found keys: {list(results.keys())}. "
                "Please ensure your test pipeline includes Pack3DDetInputs transform."
            )

        pipeline_inputs = results["inputs"]
        if "points" not in pipeline_inputs:
            available_keys = list(pipeline_inputs.keys())
            raise ValueError(
                "Expected 'points' key in pipeline inputs for CenterPoint. "
                f"Available keys: {available_keys}. "
                "For CenterPoint, voxelization is performed by the model's data_preprocessor."
            )

        points_tensor = self._to_tensor(pipeline_inputs["points"], name="points")
        if points_tensor.ndim != 2:
            raise ValueError(f"Expected points tensor with shape [N, point_features], got shape {points_tensor.shape}")

        return {"points": points_tensor}

    def get_num_samples(self) -> int:
        return len(self.data_infos)

    def get_ground_truth(self, index: int) -> Dict[str, Any]:
        sample = self.load_sample(index)

        gt_bboxes_3d = sample.get("gt_bboxes_3d", np.zeros((0, 7), dtype=np.float32))
        gt_labels_3d = sample.get("gt_labels_3d", np.zeros((0,), dtype=np.int64))

        if isinstance(gt_bboxes_3d, (list, tuple)):
            gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
        if isinstance(gt_labels_3d, (list, tuple)):
            gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)

        return {
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            "sample_idx": sample.get("sample_idx", index),
        }

    def get_class_names(self) -> List[str]:
        if hasattr(self.model_cfg, "class_names"):
            return self.model_cfg.class_names

        raise ValueError("class_names must be defined in model_cfg.")
