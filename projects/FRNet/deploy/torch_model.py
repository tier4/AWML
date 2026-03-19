"""PyTorch model wrapper for FRNet deployment.

Builds FRNet in deploy mode, loads checkpoint and runs inference.
"""

from __future__ import annotations

from time import time

import numpy as np
import numpy.typing as npt
import torch
from mmdet3d.registry import MODELS
from mmengine.config import Config
from mmengine.logging import MMLogger


class ExportModel(torch.nn.Module):
    """Deployment wrapper that exposes probabilities as the model output."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch_inputs_dict: dict, data_samples: dict | None = None) -> torch.Tensor:
        predictions = self.model(batch_inputs_dict)
        return torch.softmax(predictions["seg_logit"], dim=-1)


class TorchModel:
    """FRNet PyTorch model wrapper."""

    def __init__(self, model_cfg: Config, checkpoint_path: str) -> None:
        self.logger = MMLogger.get_current_instance()
        self.model = self._build_model(model_cfg.model, checkpoint_path)
        self.export_model = ExportModel(self.model)

    def inference(self, batch_inputs_dict: dict) -> npt.NDArray[np.float32]:
        """Forward pass, returns segmentation probabilities (N, num_classes)."""
        t_start = time()
        predictions = self.export_model(batch_inputs_dict)
        t_end = time()
        latency = np.round((t_end - t_start) * 1e3, 2)
        self.logger.info(f"Inference latency: {latency} ms")
        return predictions.cpu().detach().numpy()

    @staticmethod
    def _build_model(model_cfg: dict, checkpoint_path: str) -> torch.nn.Module:
        """Build the FRNet model in deploy mode and load weights."""
        deploy = {"deploy": True}
        model_cfg["backbone"].update(deploy)
        model_cfg["decode_head"].update(deploy)
        model = MODELS.build(model_cfg)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False)["state_dict"])
        model.eval()
        return model
