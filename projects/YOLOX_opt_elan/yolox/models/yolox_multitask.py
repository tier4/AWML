# projects/YOLOX_opt_elan/yolox/models/yolox_multitask.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.models import BaseDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import DetDataSample
from mmengine.structures import PixelData, InstanceData
from mmdet.structures import SampleList

from .heads import YOLOXSegHead  # 你的 bbox_head 和 mask_head

from mmengine.logging import print_log

@MODELS.register_module()
class YOLOXMultiTask(BaseDetector):
    """
    YOLOX MultiTask detector
    Supports bbox + mask heads.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_head=None,   # 新增 mask_head
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
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
        print_log(f"Backbone output: {[f.shape for f in x]}")
        if self.neck is not None:
            x = self.neck(x)
            print_log(f"After neck: {[f.shape for f in x]}")
        return x

    # def forward(self, imgs, **kwargs):
    #     # 训练或测试都走这里
    #     if self.training:
    #         return self.forward_train(imgs, **kwargs)
    #     else:
    #         return self.forward_test(imgs, **kwargs)

    def _forward(self, imgs, **kwargs):
        # BaseDetector 要求实现抽象方法
        return self.forward(imgs, **kwargs)
    
    def forward_train(self, imgs, gt_bboxes, gt_labels, gt_masks=None, **kwargs):
        print_log(f"Input images shape: {imgs.shape}")
        feats = self.extract_feat(imgs)
        print_log(f"Extracted features: {[f.shape for f in feats]}")
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

    def forward(self, inputs, data_samples=None, mode='tensor'):
        """Forward function with training and testing mode."""
        print("inputs.device", inputs.device)
        feats = self.extract_feat(inputs)

        if mode == 'tensor':
            # 返回预测结果 tensor
            print("Tensor mode")
            return self.bbox_head(feats)
        elif mode == 'loss':
            s = self.loss(feats, data_samples)
            print("Loss dict:", s)
            return  s # 直接返回 dict[str, Tensor]
        elif mode == 'predict':
            print("Predict mode")
            pred_instances = self.predict(inputs, data_samples)
    
            # 保留 gt_instances，让 evaluator 能访问
            for pred, data_sample in zip(pred_instances, data_samples):
                pred.gt_instances = data_sample.gt_instances
                # 如果有 gt_seg_map，也可以保留
                if hasattr(data_sample, 'gt_sem_seg'):
                    pred.gt_sem_seg = data_sample.gt_sem_seg

            # 直接返回 list[DataSample]，不要再额外套一层 []
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
            self.bbox_head.loss_by_feat(
                cls_scores, bbox_preds, objectnesses, batch_gt_instances, batch_img_metas
            )
        )

        # mask head
        if self.mask_head is not None:
            print("feats for mask head:", [f.shape for f in feats])
            seg_pred = self.mask_head(feats)
            target_size = data_samples[0].gt_sem_seg.sem_seg.shape[-2:]
            if seg_pred.shape[-2:] != target_size:
                seg_pred = F.interpolate(seg_pred, size=target_size, mode='bilinear', align_corners=False)

            print("seg_pred:", seg_pred.shape)
            gt_masks_tensor = []
            # for d in data_samples:
            #     print("d.gt_sem_seg:", d.gt_sem_seg.sem_seg.shape)
            gt_masks = torch.stack([d.gt_sem_seg.sem_seg.squeeze(0) for d in data_samples], dim=0)  # (B, H, W)
            print("seg_pred.device:", seg_pred.device)
            gt_masks = gt_masks.to(seg_pred.device)  # 和 seg_pred 同设备

            mask_loss_dict = self.mask_head.loss(seg_pred, gt_masks)  # dict: {'loss_mask': tensor}
            # 确保只包含 tensor
            for k, v in mask_loss_dict.items():
                if torch.is_tensor(v):
                    loss[k] = v
                else:
                    raise TypeError(f"mask loss '{k}' is not a tensor")

        return loss

    # def predict(self, feats, data_samples=None):
    #     """Inference for bbox + mask."""
    #     results = dict()
    #     results['bbox_results'] = self.bbox_head(feats)
    #     if self.mask_head is not None:
    #         results['mask_results'] = self.mask_head(feats)
    #     return results
    # def predict(self, feats, data_samples=None):
    #     cls_scores, bbox_preds, objectnesses = self.bbox_head(feats)
    #     batch_img_metas = ([d.metainfo for d in data_samples]
    #                     if data_samples is not None else None)

    #     pred_instances = self.bbox_head.predict_by_feat(
    #         cls_scores,
    #         bbox_preds,
    #         objectnesses,
    #         batch_img_metas=batch_img_metas,
    #         rescale=True,
    #     )

    #     seg_pred = None
    #     if self.mask_head is not None:
    #         seg_pred = self.mask_head(feats)
    #         if data_samples and seg_pred.shape[-2:] != data_samples[0].metainfo["img_shape"][:2]:
    #             seg_pred = F.interpolate(
    #                 seg_pred,
    #                 size=data_samples[0].metainfo["img_shape"][:2],
    #                 mode="bilinear",
    #                 align_corners=False,
    #             )
    #         if seg_pred is not None:
    #             seg_label_maps = seg_pred.argmax(dim=1, keepdim=True)
    #     print("seg_pred.device:", seg_pred.device)
    #     results = []
    #     for idx, instances in enumerate(pred_instances):
    #         sample = DetDataSample()
    #         if batch_img_metas:
    #             sample.set_metainfo(batch_img_metas[idx])
    #         sample.pred_instances = instances
    #         if seg_pred is not None:
    #             pixel_data = PixelData()
    #             pixel_data.data = seg_label_maps[idx]
    #             sample.pred_sem_seg = pixel_data
    #         if data_samples:
    #             if hasattr(data_samples[idx], "gt_instances"):
    #                 sample.gt_instances = data_samples[idx].gt_instances
    #             if hasattr(data_samples[idx], "gt_sem_seg"):
    #                 sample.gt_sem_seg = data_samples[idx].gt_sem_seg
    #             if hasattr(data_samples[idx], "ignored_instances"):
    #                 sample.ignored_instances = data_samples[idx].ignored_instances
    #             else:
    #                 sample.ignored_instances = InstanceData()
    #         results.append(sample)
    #         print(sample)
    #     return results
    # def predict(self,
    #             batch_inputs: Tensor,
    #             batch_data_samples: SampleList,
    #             rescale: bool = True,
    #             **kwargs) -> SampleList:
    #     """Perform forward propagation of the mask head and predict mask
    #     results on the features of the upstream network.

    #     Args:
    #         batch_inputs (Tensor): Inputs with shape (N, C, H, W).
    #         batch_data_samples (List[:obj:`DetDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
    #         rescale (bool): Whether to rescale the results.
    #             Defaults to False.

    #     Returns:
    #         list[:obj:`DetDataSample`]: Detection results of the
    #         input images. Each DetDataSample usually contain
    #         'pred_instances'. And the ``pred_instances`` usually
    #         contains following keys.

    #         - scores (Tensor): Classification scores, has a shape
    #             (num_instance, )
    #         - labels (Tensor): Labels of bboxes, has a shape
    #             (num_instances, ).
    #         - bboxes (Tensor): Has a shape (num_instances, 4),
    #             the last dimension 4 arrange as (x1, y1, x2, y2).
    #         - masks (Tensor): Has a shape (num_instances, H, W).
    #     """
    #     x = self.extract_feat(batch_inputs)
    #     if self.with_bbox:
    #         # the bbox branch does not need to be scaled to the original
    #         # image scale, because the mask branch will scale both bbox
    #         # and mask at the same time.
    #         bbox_rescale = rescale if not self.with_mask else False
    #         results_list = self.bbox_head.predict(
    #             x, batch_data_samples, rescale=bbox_rescale)
    #     else:
    #         results_list = None

    #     results_list = self.mask_head.predict(
    #         x, batch_data_samples, rescale=rescale)

    #     batch_data_samples = self.add_pred_to_datasample(
    #         batch_data_samples, results_list)
    #     return batch_data_samples

    # 在 YOLOXMultiTask 类中
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True,
                **kwargs) -> SampleList:
        
        x = self.extract_feat(batch_inputs)
        
        # 1. BBox 推理 (保持 rescale=True)
        if self.with_bbox:
            bbox_results_list = self.bbox_head.predict(x, batch_data_samples, rescale=True)
        else:
            bbox_results_list = [InstanceData() for _ in batch_data_samples]

        # 2. Mask 推理 (保持 rescale=True)
        seg_results_list = None
        if self.with_mask:
            seg_results_list = self.mask_head.predict(x, batch_data_samples, rescale=True)

        results = []
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_instances = bbox_results_list[i]
            
            # 封装预测分割
            if seg_results_list is not None:
                pixel_data = PixelData()
                pixel_data.data = seg_results_list[i]
                pixel_data.sem_seg = seg_results_list[i]
                data_sample.pred_sem_seg = pixel_data
            
            # ============================================================
            #  【关键修复】GT 数据逆变换 (Align GT to Original Image)
            # ============================================================
            
            # 获取尺寸信息
            # img_shape: Resize 后的有效尺寸 (不含 Pad)
            # ori_shape: 原图尺寸
            # scale_factor: 缩放比例 (w_scale, h_scale)
            img_h, img_w = data_sample.metainfo['img_shape']
            ori_h, ori_w = data_sample.metainfo['ori_shape']
            
            # --- 修复 A: GT BBox (除以缩放比例) ---
            if hasattr(data_sample, 'gt_instances'):
                # 也就是: boxes /= scale_factor
                # 注意：MMDet 的 Pad 默认在右下方，所以左上角坐标(0,0)不变，可以直接除
                scale_factor = data_sample.metainfo['scale_factor'] # (w_scale, h_scale)
                
                # 广播 scale_factor 到 (x1, y1, x2, y2)
                scale_factor_bbox = [scale_factor[0], scale_factor[1], scale_factor[0], scale_factor[1]]
                scale_tensor = data_sample.gt_instances.bboxes.new_tensor(scale_factor_bbox)
                
                # 执行逆变换
                data_sample.gt_instances.bboxes = data_sample.gt_instances.bboxes / scale_tensor

            # --- 修复 B: GT Mask (先 Crop 掉 Pad，再 Resize) ---
            if hasattr(data_sample, 'gt_sem_seg') and data_sample.gt_sem_seg is not None:
                gt_sem_seg_data = data_sample.gt_sem_seg.sem_seg # [H_pad, W_pad]
                
                # 1. Crop: 只取有效区域 (去除 Padding)
                # 这一步解决了"偏差"问题
                gt_valid = gt_sem_seg_data[..., :img_h, :img_w] 
                
                # 2. Resize: 插值回原图大小
                if gt_valid.shape[-2:] != (ori_h, ori_w):
                    gt_resized = F.interpolate(
                        gt_valid.unsqueeze(0).float(), # [1, 1, h, w]
                        size=(ori_h, ori_w), 
                        mode='nearest' # 类别标签必须用 nearest
                    ).squeeze(0).long()
                    
                    # 3. 替换旧容器
                    new_gt_pixel_data = PixelData()
                    new_gt_pixel_data.sem_seg = gt_resized
                    new_gt_pixel_data.data = gt_resized
                    data_sample.gt_sem_seg = new_gt_pixel_data
                elif 'data' not in data_sample.gt_sem_seg:
                    data_sample.gt_sem_seg.data = data_sample.gt_sem_seg.sem_seg

            results.append(data_sample)

        return results