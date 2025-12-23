import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

from torch import Tensor

from mmdet.models.seg_heads.base_semantic_head import BaseSemanticHead
from ..layers.network_blocks import BaseConv, CSPLayer, DWConv

from mmengine.registry import MODELS

def get_activation(name="ReLU6"):
    if name.lower() == "relu6":
        return nn.ReLU6(inplace=True)
    elif name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif name.lower() == "silu":
        return nn.SiLU(inplace=True)
    elif name.lower() == "lrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    else:
        raise AttributeError(f"Unsupported act type {name}")

@MODELS.register_module()
class YOLOXSegHead(nn.Module):
    def __init__(self, in_channels, num_classes, feat_channels=None, act_cfg=dict(type="ReLU6"), width=1.0, depthwise=False, train_cfg=None,
                 test_cfg=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.width = width
        # self.stem_channels = feat_channels if feat_channels is not None else int(64 * width)
        self.stem_channels = sum(in_channels)

        act_type = act_cfg.get("type", "ReLU6")
        self.act_fn = get_activation(act_type)

        self.train_cfg = train_cfg
        
        Conv = DWConv if depthwise else BaseConv

        # mask head layers
        self.conv1 = Conv(self.stem_channels, self.stem_channels, 3, 1, act=act_type)
        self.conv2 = Conv(self.stem_channels, self.stem_channels, 3, 1, act=act_type)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3 = Conv(self.stem_channels, self.stem_channels // 2, 3, 1, act=act_type)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv4 = Conv(self.stem_channels // 2, self.stem_channels // 2, 3, 1, act=act_type)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.out_conv = nn.Conv2d(self.stem_channels // 2, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor] or Tensor): features from backbone+neck
        Returns:
            seg_pred (Tensor): [B, num_classes, H, W]
        """
        if isinstance(feats, (list, tuple)):
            target_size = feats[0].shape[2:]
            up_feats = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in feats]
            x = torch.cat(up_feats, dim=1)  # [B, sum(C_i), H, W]
        else:
            x = feats

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up1(x)
        x = self.conv3(x)
        x = self.up2(x)
        x = self.conv4(x)
        x = self.up3(x)
        seg_pred = self.out_conv(x)
        return seg_pred

    def loss(self, seg_pred, gt_masks):
        """
        Args:
            seg_pred: [B, C, H, W]
            gt_masks: [B, H, W] long
        Returns:
            dict: {'loss_mask': ...}
        """
        return dict(loss_mask=F.cross_entropy(seg_pred, gt_masks.long(), ignore_index=255))

    def predict(self,
                x: Union[Tensor, Tuple[Tensor]],
                batch_data_samples,
                rescale: bool = False) -> List[Tensor]:
        
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        seg_preds = self.forward(x)

        input_shape = batch_img_metas[0]['batch_input_shape']
        seg_preds = F.interpolate(
            seg_preds,
            size=input_shape,
            mode='bilinear',
            align_corners=False)

        result_list = []
        for i in range(len(batch_img_metas)):
            img_meta = batch_img_metas[i]
            h, w = img_meta['img_shape']

            seg_pred = seg_preds[i][:, :h, :w]

            if rescale:
                ori_h, ori_w = img_meta['ori_shape']
                seg_pred = F.interpolate(
                    seg_pred.unsqueeze(0),
                    size=(ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False).squeeze(0)

            seg_pred = seg_pred.argmax(dim=0).to(torch.int64)
            
            result_list.append(seg_pred)

        return result_list
