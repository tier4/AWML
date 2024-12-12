from time import time

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch
from mmengine.config import Config


class OnnxModel:

    def __init__(
        self,
        deploy_cfg: Config,
        model: torch.nn.Module,
        batch_inputs_dict: dict,
        onnx_path: str,
        deploy: bool = True,
        verbose: bool = False,
    ):
        self.deploy_cfg = deploy_cfg
        self.model = model
        self.verbose = verbose
        if deploy:
            self._deploy_model(batch_inputs_dict, onnx_path)
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        self.ort_sess = ort.InferenceSession(onnx_path)

    def _deploy_model(self, batch_inputs_dict: dict, onnx_path: str) -> None:
        torch.onnx.export(
            self.model, (batch_inputs_dict, {}), onnx_path, verbose=self.verbose, **self.deploy_cfg.onnx_config
        )
        print(f"ONNX model saved to {onnx_path}.")

    def inference(self, batch_inputs_dict: dict) -> npt.ArrayLike:
        coors = batch_inputs_dict["coors"].cpu().numpy()
        points = batch_inputs_dict["points"].cpu().numpy()
        voxel_coors = batch_inputs_dict["voxel_coors"].cpu().numpy()
        inverse_map = batch_inputs_dict["inverse_map"].cpu().numpy()
        t_start = time()
        predictions = self.ort_sess.run(
            None, {"points": points, "coors": coors, "voxel_coors": voxel_coors, "inverse_map": inverse_map}
        )[0]
        t_end = time()
        latency = np.round((t_end - t_start) * 1e3, 2)
        print(f"Inference latency: {latency} ms")
        return predictions
