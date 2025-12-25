"""CenterPoint deploy-only ONNX head variants.

These heads adjust output ordering and forward behavior to improve ONNX export
and downstream inference compatibility.
"""

from typing import Dict, List, Tuple

import torch
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.registry import MODELS
from mmengine.logging import MMLogger

from projects.CenterPoint.models.dense_heads.centerpoint_head import CenterHead


@MODELS.register_module()
class SeparateHeadONNX(SeparateHead):
    """onnx support impl of mmdet3d.models.dense_heads.centerpoint_head.SeparateHead"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = MMLogger.get_current_instance()
        self._logger.info("Running SeparateHeadONNX!")

        # Note: to fix the output order
        rot_heads = {k: None for k in self.heads.keys() if "rot" in k}

        self.heads: Dict[str, None] = {
            "heatmap": None,
            "reg": None,
            "height": None,
            "dim": None,
            **rot_heads,
            "vel": None,
        }


@MODELS.register_module()
class CenterHeadONNX(CenterHead):
    """onnx support impl of mmdet3d.models.dense_heads.centerpoint_head.CenterHead"""

    def __init__(self, rot_y_axis_reference: bool = False, **kwargs):
        """
        :param switch_width_length: Set True to switch the order of width and length.
        :param rot_y_axis_reference: Set True to output rotation of sin(y), cos(x) relative to the
        y-axis.
        """
        super().__init__(**kwargs)

        assert len(self.task_heads) == 1, "CenterPoint must use a single-task head"
        self.task_heads: List[SeparateHeadONNX]
        self.output_names: List[str] = list(self.task_heads[0].heads.keys())
        self._logger = MMLogger.get_current_instance()
        self._rot_y_axis_reference = rot_y_axis_reference
        self._logger.info(f"Running CenterHeadONNX! Output rotations in y-axis: {self._rot_y_axis_reference}")

    def _export_forward_rot_y_axis_reference(self, head_tensors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        TODO(KokSeang): This is a dirty and quick fix, we need to add the same operation to all
        outputs to prevent reordering from ONNX export. However, we probably should use onnx_graphsurgeon
        to modify them manually.
        """
        # Heatmap
        heatmap_tensors = head_tensors["heatmap"][:, torch.tensor([0, 1, 2, 3, 4], dtype=torch.int), :, :]
        heatmap_scale_factors = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(device=heatmap_tensors.device)
        heatmap_scale_factors = heatmap_scale_factors.view([1, -1, 1, 1])
        scale_heatmap_tensors = torch.mul(heatmap_tensors, heatmap_scale_factors)
        # Reg
        reg_tensors = head_tensors["reg"][:, torch.tensor([0, 1], dtype=torch.int), :, :]
        reg_scale_factors = torch.tensor([1.0, 1.0]).to(device=reg_tensors.device)
        reg_scale_factors = reg_scale_factors.view([1, -1, 1, 1])
        scale_reg_tensors = torch.mul(reg_tensors, reg_scale_factors)

        # Height
        height_tensors = head_tensors["height"][:, torch.tensor([0], dtype=torch.int), :, :]
        height_scale_factors = torch.tensor([1.0]).to(device=height_tensors.device)
        height_scale_factors = height_scale_factors.view([1, -1, 1, 1])
        scale_height_tensors = torch.mul(height_tensors, height_scale_factors)

        # Dim
        # Swap length, width, height to width, length, height
        flip_dim_tensors = head_tensors["dim"][:, torch.tensor([1, 0, 2], dtype=torch.int), :, :]
        dim_scale_factors = torch.tensor([1.0, 1.0, 1.0]).to(device=flip_dim_tensors.device)
        dim_scale_factors = dim_scale_factors.view([1, -1, 1, 1])
        scale_flip_dim_tensors = torch.mul(flip_dim_tensors, dim_scale_factors)

        # Rot
        # Swap sin(y), cos(x) to cos(x), sin(y)
        flip_rot_tensors = head_tensors["rot"][:, torch.tensor([1, 0], dtype=torch.int), :, :]
        # Negate -cos(x) and -sin(y) to change direction
        rot_scale_factors = torch.tensor([-1.0, -1.0]).to(device=flip_rot_tensors.device)
        rot_scale_factors = rot_scale_factors.view([1, -1, 1, 1])
        scale_flip_rot_tensors = torch.mul(flip_rot_tensors, rot_scale_factors)

        # Vel
        vel_tensors = head_tensors["vel"][:, torch.tensor([0, 1], dtype=torch.int), :, :]
        vel_scale_factors = torch.tensor([1.0, 1.0]).to(device=vel_tensors.device)
        vel_scale_factors = vel_scale_factors.view([1, -1, 1, 1])
        scale_vel_tensors = torch.mul(vel_tensors, vel_scale_factors)

        return (
            scale_heatmap_tensors,
            scale_reg_tensors,
            scale_height_tensors,
            scale_flip_dim_tensors,
            scale_flip_rot_tensors,
            scale_vel_tensors,
        )

    def _export_forward_single(self, head_tensors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Forward normally that uses the x-axis reference.
        """
        ret_list: List[torch.Tensor] = [head_tensors[head_name] for head_name in self.output_names]
        return tuple(ret_list)

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward pass.
        Args:
            x (List[torch.Tensor]): multi-level features
        Returns:
            pred (Tuple[torch.Tensor]): Output results for tasks.
        """
        assert len(x) == 1, "The input of CenterHeadONNX must be a single-level feature"
        x = self.shared_conv(x[0])
        head_tensors: Dict[str, torch.Tensor] = self.task_heads[0](x)
        if self._rot_y_axis_reference:
            return self._export_forward_rot_y_axis_reference(head_tensors=head_tensors)
        else:
            return self._export_forward_single(head_tensors=head_tensors)
