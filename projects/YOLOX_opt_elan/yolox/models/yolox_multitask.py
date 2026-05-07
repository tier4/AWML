import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from .heads import YOLOXSegHead


@MODELS.register_module()
class YOLOXMultiTask(BaseDetector):
    """
    YOLOX MultiTask detector
    Supports bbox + mask heads.
    """

    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        mask_head=None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = MODELS.build(bbox_head)
        if mask_head is not None:
            mask_head.update(train_cfg=train_cfg)
            mask_head.update(test_cfg=test_cfg)
            self.mask_head = MODELS.build(mask_head) if mask_head is not None else None
        self.data_preprocessor = MODELS.build(data_preprocessor) if data_preprocessor else None

    def extract_feat(self, inputs):
        x = self.backbone(inputs)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def _forward(self, imgs, **kwargs):
        return self.forward(imgs, **kwargs)

    def forward_train(self, imgs, gt_bboxes, gt_labels, gt_masks=None, **kwargs):
        feats = self.extract_feat(imgs)
        losses = dict()
        losses.update(self.bbox_head.loss(feats[-1], gt_bboxes, gt_labels))
        if self.mask_head is not None and gt_masks is not None:
            mask_pred = self.mask_head(feats)
            losses.update(self.mask_head.loss(mask_pred, gt_masks))
        return losses

    def forward_test(self, imgs, **kwargs):
        feats = self.extract_feat(imgs)
        bbox_results = self.bbox_head(feats[-1])
        mask_results = None
        if self.mask_head is not None:
            mask_results = self.mask_head(feats)
        return dict(bboxes=bbox_results, masks=mask_results)

    def forward(self, inputs, data_samples=None, mode="tensor"):
        """Forward function with training and testing mode."""
        feats = self.extract_feat(inputs)

        if mode == "tensor":
            return self.bbox_head(feats)
        elif mode == "loss":
            s = self.loss(feats, data_samples)
            return s
        elif mode == "predict":
            pred_instances = self.predict(inputs, data_samples)

            for pred, data_sample in zip(pred_instances, data_samples):
                pred.gt_instances = data_sample.gt_instances
                if hasattr(data_sample, "gt_sem_seg"):
                    pred.gt_sem_seg = data_sample.gt_sem_seg

            return pred_instances
        else:
            raise ValueError(f"Invalid mode {mode}")

    def loss(self, feats, data_samples):
        loss = dict()
        # bbox head forward
        cls_scores, bbox_preds, objectnesses = self.bbox_head(feats)
        batch_gt_instances = [d.gt_instances for d in data_samples]
        batch_img_metas = [d.metainfo for d in data_samples]

        loss.update(
            self.bbox_head.loss_by_feat(cls_scores, bbox_preds, objectnesses, batch_gt_instances, batch_img_metas)
        )

        # mask head
        if self.mask_head is not None:
            seg_pred = self.mask_head(feats)
            target_size = data_samples[0].gt_sem_seg.sem_seg.shape[-2:]
            if seg_pred.shape[-2:] != target_size:
                seg_pred = F.interpolate(seg_pred, size=target_size, mode="bilinear", align_corners=False)

            gt_masks_tensor = []
            gt_masks = torch.stack([d.gt_sem_seg.sem_seg.squeeze(0) for d in data_samples], dim=0)  # (B, H, W)
            gt_masks = gt_masks.to(seg_pred.device)

            mask_loss_dict = self.mask_head.loss(seg_pred, gt_masks)
            for k, v in mask_loss_dict.items():
                if torch.is_tensor(v):
                    loss[k] = v
                else:
                    raise TypeError(f"mask loss '{k}' is not a tensor")

        return loss

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True, **kwargs
    ) -> SampleList:

        x = self.extract_feat(batch_inputs)

        if self.with_bbox:
            bbox_results_list = self.bbox_head.predict(x, batch_data_samples, rescale=True)
        else:
            bbox_results_list = [InstanceData() for _ in batch_data_samples]

        seg_results_list = None
        if self.with_mask:
            seg_results_list = self.mask_head.predict(x, batch_data_samples, rescale=True)

        results = []
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_instances = bbox_results_list[i]

            if seg_results_list is not None:
                pixel_data = PixelData()
                pixel_data.data = seg_results_list[i]
                pixel_data.sem_seg = seg_results_list[i]
                data_sample.pred_sem_seg = pixel_data

            img_h, img_w = data_sample.metainfo["img_shape"]
            ori_h, ori_w = data_sample.metainfo["ori_shape"]

            if hasattr(data_sample, "gt_instances"):

                scale_factor = data_sample.metainfo["scale_factor"]  # (w_scale, h_scale)

                scale_factor_bbox = [scale_factor[0], scale_factor[1], scale_factor[0], scale_factor[1]]
                scale_tensor = data_sample.gt_instances.bboxes.new_tensor(scale_factor_bbox)

                data_sample.gt_instances.bboxes = data_sample.gt_instances.bboxes / scale_tensor

            if hasattr(data_sample, "gt_sem_seg") and data_sample.gt_sem_seg is not None:
                gt_sem_seg_data = data_sample.gt_sem_seg.sem_seg  # [H_pad, W_pad]

                gt_valid = gt_sem_seg_data[..., :img_h, :img_w]

                if gt_valid.shape[-2:] != (ori_h, ori_w):
                    gt_resized = (
                        F.interpolate(
                            gt_valid.unsqueeze(0).float(), size=(ori_h, ori_w), mode="nearest"  # [1, 1, h, w]
                        )
                        .squeeze(0)
                        .long()
                    )

                    new_gt_pixel_data = PixelData()
                    new_gt_pixel_data.sem_seg = gt_resized
                    new_gt_pixel_data.data = gt_resized
                    data_sample.gt_sem_seg = new_gt_pixel_data
                elif "data" not in data_sample.gt_sem_seg:
                    data_sample.gt_sem_seg.data = data_sample.gt_sem_seg.sem_seg

            results.append(data_sample)

        return results
