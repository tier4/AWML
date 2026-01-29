from time import time
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from mmdet3d.registry import MODELS
from mmengine.config import Config


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
        if hasattr(model_cfg, "class_mapping"):
            reverse_class_mapping: Dict[int, str] = {}
            for class_name, idx in model_cfg.class_mapping.items():
                if idx == model_cfg.ignore_index:
                    continue
                if reverse_class_mapping.get(idx, None) is None:
                    reverse_class_mapping[idx] = [class_name]
                else:
                    reverse_class_mapping[idx].append(class_name)

            class_names: List[str] = [None] * len(reverse_class_mapping)
            for idx, classes in reverse_class_mapping.items():
                joined_classes = "+".join(classes)
                class_names[idx] = joined_classes

            return class_names

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
