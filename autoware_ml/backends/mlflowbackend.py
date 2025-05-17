import re

import mlflow
from mmengine.config import Config, ConfigDict
from mmengine.registry import VISBACKENDS
from mmengine.visualization import MLflowVisBackend
from mmengine.visualization.vis_backend import force_init_env


def sanitize_mlflow_key(key: str) -> str:
    """
    Sanitize a string key to comply with MLflow's allowed character set.

    Allowed characters:
        - alphanumerics (a-z, A-Z, 0-9)
        - underscore (_)
        - dash (-)
        - period (.)
        - space ( )
        - colon (:)
        - slash (/)

    Any disallowed characters are replaced with an underscore (_).

    Args:
        key (str): The original key name.

    Returns:
        str: The sanitized key name.
    """
    return re.sub(r"[^a-zA-Z0-9_\-.:/ ]", "_", key)


@VISBACKENDS.register_module()
class SafeMLflowVisBackend(MLflowVisBackend):
    """Safe MLflow Visualization Backend that sanitizes keys before logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to mlflow.

        Args:
            config (Config): The Config object
        """
        self.cfg = config
        if self._tracked_config_keys is None:
            raw_dict = self._flatten(self.cfg.to_dict())
        else:
            tracked_cfg = {k: self.cfg[k] for k in self._tracked_config_keys}
            raw_dict = self._flatten(tracked_cfg)

        sanitized_params = {}
        for key, value in raw_dict.items():
            safe_key = sanitize_mlflow_key(key)
            sanitized_params[safe_key] = value

        self._mlflow.log_params(sanitized_params)
        self._mlflow.log_text(self.cfg.pretty_text, "config.py")
