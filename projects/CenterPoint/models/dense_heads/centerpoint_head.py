from typing import List, Tuple

import torch
from mmdet3d.models import CenterHead as _CenterHead
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.registry import MODELS
from mmengine import print_log
from mmengine.structures import InstanceData


@MODELS.register_module(force=True)
class CustomSeparateHead(SeparateHead):

    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        init_bias=-2.19,
        conv_cfg=dict(type="Conv2d"),
        norm_cfg=dict(type="BN2d"),
        bias="auto",
        init_cfg=None,
        **kwargs,
    ):
        """Overwritten class of SeparateHead to fix the initialization of bias weights."""
        assert init_cfg is None, "To prevent abnormal initialization " "behavior, init_cfg is not allowed to be set"
        super(CustomSeparateHead, self).__init__(
            in_channels=in_channels,
            heads=heads,
            head_conv=head_conv,
            final_kernel=final_kernel,
            init_bias=init_bias,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
            init_cfg=init_cfg,
            **kwargs,
        )
        if init_cfg is None:
            self.init_cfg = dict(type="Kaiming", layer="Conv2d")

        self.init_bias_weights()

    def init_bias_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == "heatmap":
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)


@MODELS.register_module(force=True)
class CenterHead(_CenterHead):
    """overwritten class of CenterHead
    Note:
        We add class-wise loss implementation.
    TODO(KokSeang):
        We still using `loss_bbox` in this implementation, loss_reg, loss_height, loss_dim,
        loss_rot and loss_vel will be implemented in the next version.
        For this reason, we need to set `code_weights` for each loss in the config `model.train_cfg`.
    """

    def __init__(
        self,
        freeze_shared_conv: bool = False,
        freeze_task_heads: bool = False,
        **kwargs,
    ):
        super(CenterHead, self).__init__(**kwargs)
        loss_cls = kwargs["loss_cls"]
        self._class_wise_loss = loss_cls.get("reduction") == "none"
        if not self._class_wise_loss:
            print_log("If you want to see a class-wise heatmap loss, use reduction='none' of 'loss_cls'.")

        self.freeze_shared_conv = freeze_shared_conv
        self.freeze_task_heads = freeze_task_heads
        self._freeze_parameters()

    def _freeze_parameters(self) -> None:
        """Freeze parameters in the head."""
        if self.freeze_shared_conv:
            print_log("Freeze shared conv")
            for params in self.shared_conv.parameters():
                params.requires_grad = False

        if self.freeze_task_heads:
            print_log("Freeze task heads")
            for task in self.task_heads:
                for params in task.parameters():
                    params.requires_grad = False

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]], batch_gt_instances_3d: List[InstanceData], *args, **kwargs):
        """Loss function for CenterHead.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and\
                ``labels_3d`` attributes.

        Returns:
            dict[str,torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(batch_gt_instances_3d)
        loss_dict = dict()

        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]["heatmap"] = clip_sigmoid(preds_dict[0]["heatmap"])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            class_names: List[str] = self.class_names[task_id]

            if self._class_wise_loss:
                loss_heatmap_cls: torch.Tensor = self.loss_cls(
                    preds_dict[0]["heatmap"],
                    heatmaps[task_id],
                )
                loss_heatmap_cls = loss_heatmap_cls.sum((0, 2, 3)) / max(num_pos, 1)
                for cls_i, class_name in enumerate(class_names):
                    loss_dict[f"task{task_id}.loss_heatmap_{class_name}"] = loss_heatmap_cls[cls_i]
            else:
                loss_heatmap = self.loss_cls(preds_dict[0]["heatmap"], heatmaps[task_id], avg_factor=max(num_pos, 1))
                loss_dict[f"task{task_id}.loss_heatmap"] = loss_heatmap

            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]["anno_box"] = torch.cat(
                (
                    preds_dict[0]["reg"],
                    preds_dict[0]["height"],
                    preds_dict[0]["dim"],
                    preds_dict[0]["rot"],
                    preds_dict[0]["vel"],
                ),
                dim=1,
            )

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]["anno_box"].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get("code_weights", None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f"task{task_id}.loss_bbox"] = loss_bbox
        return loss_dict
