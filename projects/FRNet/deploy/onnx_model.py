import sys
from time import time

import numpy.typing as npt
import numpy as np
import onnxruntime as ort
import onnx
import torch

sys.path.append('/workspace/projects/FRNet')
from frnet.models.segmentors.frnet import FRNet
from configs.deploy.frnet_tensorrt_dynamic import onnx_config


class OnnxModel:

    def __init__(self,
                 model: FRNet,
                 batch_inputs_dict: dict,
                 onnx_path: str,
                 deploy: bool = True,
                 verbose: bool = False):
        self.model = model
        self.verbose = verbose
        if deploy:
            self._deploy_model(batch_inputs_dict, onnx_path)
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        self.ort_sess = ort.InferenceSession(onnx_path)

    def _deploy_model(self, batch_inputs_dict: dict, onnx_path: str) -> None:
        torch.onnx.export(
            self.model,
            (batch_inputs_dict, {}),
            onnx_path,
            verbose=self.verbose,
            **onnx_config)
        print(f'ONNX model saved to {onnx_path}.')

    def inference(self, batch_inputs_dict: dict) -> npt.ArrayLike:
        coors = batch_inputs_dict['coors'].cpu().numpy()
        points = batch_inputs_dict['points'].cpu().numpy()
        voxel_coors = batch_inputs_dict['voxel_coors'].cpu().numpy()
        inverse_map = batch_inputs_dict['inverse_map'].cpu().numpy()
        t_start = time()
        predictions = self.ort_sess.run(
            None, {
                'points': points,
                'coors': coors,
                'voxel_coors': voxel_coors,
                'inverse_map': inverse_map
            })[0]
        t_end = time()
        latency = np.round((t_end - t_start) * 1e3, 2)
        print(f'Inference latency: {latency} ms')
        return predictions
