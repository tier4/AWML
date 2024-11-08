import sys
from time import time

import numpy as np
import numpy.typing as npt
import torch

from utils import load_model_cfg

sys.path.append('/workspace/projects/FRNet')
from frnet.models.segmentors.frnet import FRNet


class TorchModel:

    def __init__(self, model_path: str, checkpoint_path: str):
        cfg, self.class_names = load_model_cfg(model_path)
        self.model = self._build_model(cfg, checkpoint_path)

    def _build_model(self, model_cfg: dict, checkpoint_path: str) -> FRNet:
        deploy = {'deploy': True}
        model_cfg['backbone'].update(deploy)
        model_cfg['decode_head'].update(deploy)
        model = FRNet(
            voxel_encoder=model_cfg['voxel_encoder'],
            backbone=model_cfg['backbone'],
            decode_head=model_cfg['decode_head'],
            auxiliary_head=model_cfg['auxiliary_head'],
            data_preprocessor=model_cfg['data_preprocessor'])
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        model.eval()
        return model

    def inference(self, batch_inputs_dict: dict) -> npt.ArrayLike:
        t_start = time()
        predictions = self.model(batch_inputs_dict)
        t_end = time()
        latency = np.round((t_end - t_start) * 1e3, 2)
        print(f'Inference latency: {latency} ms')
        return predictions['seg_logit'].cpu().detach().numpy()
