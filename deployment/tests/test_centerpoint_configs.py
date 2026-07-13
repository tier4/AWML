"""Parse-validation for every CenterPoint deploy config.

Loads each ``deployment/projects/centerpoint/config/deploy_config*.py`` and feeds its
sections through the typed parsers in ``deployment.config.schema``. This catches
structural/rename errors (e.g. wrong component keys, ``num_warmup_samples`` leftovers,
invalid precision_policy/scenarios) WITHOUT needing a real checkpoint, CUDA, or a full
``BaseDeploymentConfig`` (which validates checkpoint existence + CUDA availability).

Requires CPU torch + mmengine (the deployment runtime image); no GPU needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mmengine.config import Config

from deployment.config.schema import (
    ComponentsConfig,
    DeviceConfig,
    EvaluationConfig,
    ExportConfig,
    OnnxConfig,
    TensorRTConfig,
    VerificationConfig,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "projects" / "centerpoint" / "config"
_CONFIG_FILES = sorted(_CONFIG_DIR.glob("deploy_config*.py"))


def _config_id(path: Path) -> str:
    return path.stem


@pytest.mark.parametrize("config_path", _CONFIG_FILES, ids=[_config_id(p) for p in _CONFIG_FILES])
def test_centerpoint_config_parses(config_path: Path) -> None:
    """Every CenterPoint deploy config parses cleanly under the NEW typed schema."""
    cfg = Config.fromfile(str(config_path))

    # Required sections feed the typed parsers (these raise on structural errors).
    components_cfg = ComponentsConfig.from_dict(cfg["components"])
    ExportConfig.from_dict(cfg["export"])
    OnnxConfig.from_dict(cfg.get("onnx_config"))
    TensorRTConfig.from_dict(cfg.get("tensorrt_config", {}))
    EvaluationConfig.from_dict(cfg.get("evaluation", {}))
    VerificationConfig.from_dict(cfg.get("verification", {}))
    DeviceConfig.from_dict(cfg.get("devices", {}))

    # Component keys must be the canonical CenterPoint ids (catches un-renamed outer keys).
    # This is the same required-component set that CenterPointDeploymentConfig._validate_components
    # enforces at config-construction time.
    component_names = set(components_cfg.component_names())
    assert component_names == {
        "pts_voxel_encoder",
        "pts_backbone_neck_head",
    }, f"{config_path.name}: unexpected component keys {sorted(component_names)}"


def test_config_dir_is_nonempty() -> None:
    """Guard against the glob silently matching nothing."""
    assert _CONFIG_FILES, f"No deploy_config*.py found under {_CONFIG_DIR}"
