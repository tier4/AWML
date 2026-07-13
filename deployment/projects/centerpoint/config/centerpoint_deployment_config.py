"""CenterPoint-specific deployment config.

Extends :class:`~deployment.config.base.BaseDeploymentConfig` to model the CenterPoint-only
deploy-config keys as typed attributes, so the entrypoint and export pipeline never reach
back into the raw MMEngine ``Config``. This is the typed home for the keys the generic
sections intentionally do not model.
"""

from __future__ import annotations

from mmengine.config import Config

from deployment.config.base import BaseDeploymentConfig


class CenterPointDeploymentConfig(BaseDeploymentConfig):
    """Deployment config for CenterPoint.

    Adds typed attributes for the CenterPoint-only deploy-config keys:

    - ``rot_y_axis_reference``: output rotation as ``sin(y), cos(x)`` relative to the y-axis in the
      exported head, matching the ONNX-compatible output format expected by the runtime
      (default ``False``).
    """

    #: Components CenterPoint always splits into for multi-file ONNX/TensorRT export.
    _REQUIRED_COMPONENTS = ("pts_voxel_encoder", "pts_backbone_neck_head")

    def __init__(self, deploy_cfg: Config) -> None:
        super().__init__(deploy_cfg)
        self.rot_y_axis_reference: bool = bool(deploy_cfg.get("rot_y_axis_reference", False))
        self._validate_components()

    def _validate_components(self) -> None:
        """Fail early if the deploy config is missing a required CenterPoint component.

        Validated here (rather than via the project registry) so both CenterPoint and BEVFusion
        check their component layout the same way — at config construction time.
        """
        for component_name in self._REQUIRED_COMPONENTS:
            self.components_cfg.get_component(component_name)
