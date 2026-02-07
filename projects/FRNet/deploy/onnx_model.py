"""ONNX model wrapper for FRNet deployment.

Exports a PyTorch model to ONNX and runs inference via ONNX Runtime.
"""

from __future__ import annotations

from time import time

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger


class OnnxModel:
    """FRNet ONNX model wrapper.

    Optionally exports the model to ONNX on construction (when deploy=True),
    then validates and loads it into an ONNX Runtime session.
    """

    def __init__(
        self,
        deploy_cfg: Config,
        model: torch.nn.Module,
        batch_inputs_dict: dict,
        onnx_path: str,
        deploy: bool = True,
        verbose: bool = False,
    ) -> None:
        self._deploy_cfg = deploy_cfg
        self._verbose = verbose
        self.logger = MMLogger.get_current_instance()

        if deploy:
            self._export(model, batch_inputs_dict, onnx_path)

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        self._session = ort.InferenceSession(onnx_path)

    def inference(self, batch_inputs_dict: dict) -> npt.NDArray[np.float32]:
        """Run ONNX Runtime inference, returns logits (N, num_classes)."""
        coors = batch_inputs_dict["coors"].cpu().numpy()
        points = batch_inputs_dict["points"].cpu().numpy()
        voxel_coors = batch_inputs_dict["voxel_coors"].cpu().numpy()
        inverse_map = batch_inputs_dict["inverse_map"].cpu().numpy()

        t_start = time()
        predictions = self._session.run(
            None,
            {
                "points": points,
                "coors": coors,
                "voxel_coors": voxel_coors,
                "inverse_map": inverse_map,
            },
        )[0]
        t_end = time()

        latency = np.round((t_end - t_start) * 1e3, 2)
        self.logger.info(f"Inference latency: {latency} ms")
        return predictions

    def _export(self, model: torch.nn.Module, batch_inputs_dict: dict, onnx_path: str) -> None:
        """Export the PyTorch model to ONNX."""
        torch.onnx.export(
            model,
            (batch_inputs_dict, {}),
            onnx_path,
            verbose=self._verbose,
            **self._deploy_cfg.onnx_config,
        )
        self.logger.info(f"ONNX model saved to {onnx_path}.")
