from typing import List, Tuple

import torch
from mmdet3d.models import CenterHead as _CenterHead
from mmdet3d.models import circle_nms
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.models.layers import nms_bev
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.registry import MODELS
from mmengine import print_log
from mmengine.structures import InstanceData
from mmdet3d.structures import Det3DDataSample, xywhr2xyxyr


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

    def predict(
        self, pts_feats: dict, batch_data_samples: List[Det3DDataSample], rescale: bool = True, **kwargs
    ) -> List[InstanceData]:
        """Override predict to attach per-box class probability vectors."""

        preds_dict = self(pts_feats)
        batch_input_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        return self.predict_by_feat(preds_dict, batch_input_metas, rescale=rescale, **kwargs)

    def predict_by_feat(
        self,
        preds_dicts: Tuple[List[dict]],
        batch_input_metas: List[dict],
        *args,
        **kwargs,
    ) -> List[InstanceData]:
        """Generate bboxes from bbox head predictions and keep class score vectors."""
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]["heatmap"].shape[0]
            raw_heatmap = preds_dict[0]["heatmap"]
            batch_heatmap = raw_heatmap.sigmoid()

            batch_reg = preds_dict[0]["reg"]
            batch_hei = preds_dict[0]["height"]

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]["dim"])
            else:
                batch_dim = preds_dict[0]["dim"]

            batch_rots = preds_dict[0]["rot"][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]["rot"][:, 1].unsqueeze(1)

            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"]
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id,
                raw_heatmap=raw_heatmap,
            )
            assert self.test_cfg["nms_type"] in ["circle", "rotate"]
            batch_reg_preds = [box["bboxes"] for box in temp]
            batch_cls_preds = [box["scores"] for box in temp]
            batch_cls_labels = [box["labels"] for box in temp]
            batch_cls_vectors = [box.get("class_scores") for box in temp]
            if self.test_cfg["nms_type"] == "circle":
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]["bboxes"]
                    scores = temp[i]["scores"]
                    labels = temp[i]["labels"]
                    class_scores = batch_cls_vectors[i]
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg["min_radius"][task_id],
                            post_max_size=self.test_cfg["post_max_size"],
                        ),
                        dtype=torch.long,
                        device=boxes.device,
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    if class_scores is not None:
                        ret["class_scores"] = class_scores[keep]
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(
                        num_class_with_bg,
                        batch_cls_preds,
                        batch_reg_preds,
                        batch_cls_labels,
                        batch_cls_vectors,
                        batch_input_metas,
                    )
                )

        # Merge branches results
        num_samples = len(rets[0])
        total_classes = sum(self.num_classes)

        ret_list = []
        for i in range(num_samples):
            bboxes_list = []
            scores_list = []
            labels_list = []
            class_scores_list = []
            label_offset = 0
            for task_idx, ret in enumerate(rets):
                task_res = ret[i]
                bboxes_list.append(task_res["bboxes"])
                scores_list.append(task_res["scores"])
                labels_list.append(task_res["labels"] + label_offset)
                if "class_scores" in task_res:
                    cs = task_res["class_scores"]
                    if cs.numel() > 0:
                        padded = cs.new_zeros(cs.size(0), total_classes)
                        padded[:, label_offset : label_offset + self.num_classes[task_idx]] = cs
                    else:
                        padded = cs.new_zeros((0, total_classes))
                    class_scores_list.append(padded)
                label_offset += self.num_classes[task_idx]

            bboxes = torch.cat(bboxes_list)
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = batch_input_metas[i]["box_type_3d"](bboxes, self.bbox_coder.code_size)
            scores = torch.cat(scores_list)
            labels = torch.cat(labels_list).int()

            temp_instances = InstanceData()
            temp_instances.bboxes_3d = bboxes
            temp_instances.scores_3d = scores
            temp_instances.labels_3d = labels
            if class_scores_list:
                temp_instances.class_scores_3d = torch.cat(class_scores_list, dim=0)
            ret_list.append(temp_instances)
        return ret_list

    def get_task_detections(
        self,
        num_class_with_bg,
        batch_cls_preds,
        batch_reg_preds,
        batch_cls_labels,
        batch_cls_vectors,
        img_metas,
    ):
        """Rotate nms for each task with class score vectors."""
        predictions_dicts = []
        post_center_range = self.test_cfg["post_center_limit_range"]
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range, dtype=batch_reg_preds[0].dtype, device=batch_reg_preds[0].device
            )

        for i, (box_preds, cls_preds, cls_labels, cls_vectors) in enumerate(
            zip(batch_reg_preds, batch_cls_preds, batch_cls_labels, batch_cls_vectors)
        ):

            # Apply NMS in bird eye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(cls_preds.shape[0], device=cls_preds.device, dtype=torch.long)
            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg["score_threshold"] > 0.0:
                thresh = torch.tensor([self.test_cfg["score_threshold"]], device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)
                cls_vectors = cls_vectors[top_scores_keep]

            if top_scores.shape[0] != 0:
                if self.test_cfg["score_threshold"] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(img_metas[i]["box_type_3d"](box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg["nms_thr"],
                    pre_max_size=self.test_cfg["pre_max_size"],
                    post_max_size=self.test_cfg["post_max_size"],
                )
            else:
                selected = []

            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]
            selected_cls_vectors = cls_vectors[selected] if cls_vectors is not None else None

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                final_box_preds = selected_boxes
                final_scores = selected_scores
                final_labels = selected_labels
                final_cls_vectors = selected_cls_vectors
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask],
                        class_scores=final_cls_vectors[mask] if final_cls_vectors is not None else None,
                    )
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels,
                        class_scores=final_cls_vectors,
                    )
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                empty_scores = torch.zeros([0], dtype=dtype, device=device)
                empty_labels = torch.zeros([0], dtype=top_labels.dtype, device=device)
                empty_cls = torch.zeros([0, num_class_with_bg], dtype=dtype, device=device)
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size], dtype=dtype, device=device),
                    scores=empty_scores,
                    labels=empty_labels,
                    class_scores=empty_cls,
                )

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
