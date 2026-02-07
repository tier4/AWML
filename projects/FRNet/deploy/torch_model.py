from time import time
from typing import List

import numpy as np
import numpy.typing as npt
import torch
from mmdet3d.registry import MODELS
from mmengine.config import Config

from autoware_ml.segmentation3d.datasets.utils import class_mapping_to_names


class TorchModel:

    def __init__(
        self,
        deploy_cfg: Config,
        model_cfg: Config,
        checkpoint_path: str,
    ):
        self.class_names = self.get_class_names(model_cfg)
        self.model = self._build_model(model_cfg.model, checkpoint_path)

    def get_class_names(self, model_cfg: Config) -> List[str]:
        # nuScenes
        if hasattr(model_cfg, "class_names"):
            return model_cfg.class_names
        # T4dataset
        elif hasattr(model_cfg, "class_mapping"):
            ignore_index = getattr(model_cfg, "ignore_index", -1)
            return class_mapping_to_names(model_cfg.class_mapping, ignore_index)
        else:
            raise KeyError("Class names or class mapping not found in model config.")

    def _build_model(self, model_cfg: dict, checkpoint_path: str) -> "FRNet":
        deploy = {"deploy": True}
        model_cfg["backbone"].update(deploy)
        model_cfg["decode_head"].update(deploy)
        model = MODELS.build(model_cfg)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False)["state_dict"])
        model.eval()
        return model

    def inference(self, batch_inputs_dict: dict) -> npt.ArrayLike:
        t_start = time()
        predictions = self.model(batch_inputs_dict)
        t_end = time()
        latency = np.round((t_end - t_start) * 1e3, 2)
        print(f"Inference latency: {latency} ms")
        return predictions["seg_logit"].cpu().detach().numpy()
