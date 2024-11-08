from copy import copy
import importlib
import os
import sys

from typing import List, Tuple

sys.path.append('/workspace/projects/FRNet')
from configs.nuscenes.model.frnet import model as BaseConfig


def _get_module(model_path: str) -> str:
    path_parts = model_path.split(os.sep)
    index = path_parts.index('configs')
    module_name = os.path.splitext(
        os.path.join(*path_parts[index:]))[0].replace(os.sep, '.')
    module = importlib.import_module(module_name)

    return module


def load_model_cfg(model_path: str) -> Tuple[dict, List[str]]:
    module = _get_module(model_path)
    model_config = module.model
    config = copy(BaseConfig)

    def merge_dicts(config1: dict, config2: dict):
        for key, value in config2.items():
            if key in config1:
                if isinstance(config1[key], dict) and isinstance(value, dict):
                    merge_dicts(config1[key], value)
                else:
                    config1[key] = value
            else:
                config1[key] = value

    merge_dicts(config, model_config)

    class_names = module.class_names

    return config, class_names


def load_preprocessing_cfg(model_path: str) -> dict:
    module = _get_module(model_path)

    model_config = module.model
    train_config = module.train_pipeline

    frustum_region_group = model_config['data_preprocessor']
    range_interpolation = [
        d for d in train_config if d['type'] == 'RangeInterpolation'
    ][0]
    range_interpolation.pop('type')

    return {
        'frustum_region_group': frustum_region_group,
        'range_interpolation': range_interpolation
    }
