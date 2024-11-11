from time import time

import numpy as np
import numpy.typing as npt
import torch
from mmdet3d.registry import MODELS
from mmengine.config import Config


class TorchModel:

    def __init__(self, deploy_cfg: Config, model_cfg: Config,
                 checkpoint_path: str):
        self.class_names = model_cfg.class_names
        self.model = self._build_model(model_cfg.model, checkpoint_path)

    def _build_model(self, model_cfg: dict, checkpoint_path: str) -> 'FRNet':
        deploy = {'deploy': True}
        model_cfg['backbone'].update(deploy)
        model_cfg['decode_head'].update(deploy)
        model = MODELS.build(model_cfg)
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
