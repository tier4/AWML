from copy import deepcopy
from typing import List, Tuple

import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet3d.models.utils import clip_sigmoid, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.registry import MODELS
from mmengine import print_log
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor, nn

from projects.CenterPoint.models.dense_heads.centerpoint_head import CenterHead


@MODELS.register_module()
class MultiHeadSeparateHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int, optional): Output channels.
            Default: 64.
        final_kernel (int, optional): Kernel size for the last conv layer.
            Default: 1.
        init_bias (float, optional): Initial bias. Default: -2.19.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        init_bias=-2.19,
        conv_cfg=dict(type="Conv2d"),
        norm_cfg=dict(type="BN2d"),
        bias="auto",
        init_cfg=None,
        **kwargs,
    ):
        assert init_cfg is None, "To prevent abnormal initialization " "behavior, init_cfg is not allowed to be set"
        super(MultiHeadSeparateHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, convs = self.heads[head]
            num_conv = len(convs)
            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                stride, padding, kernel_size = convs[i]
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                    )
                )
                c_in = head_conv

            stride, padding, kernel_size = convs[-1]
            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=True,
                )
            )
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type="Kaiming", layer="Conv2d")

        self.init_bias_weights()

    def init_bias_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == "heatmap":
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == "heatmap":
                print_log("Initialize heatmap bias")
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg (torch.Tensor): 2D regression value with the
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@MODELS.register_module(force=True)
class MultiHeadCenterHead(CenterHead):
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
        detection_heads,
        freeze_shared_conv: bool = False,
        freeze_task_heads: bool = False,
        **kwargs,
    ):
        super(MultiHeadCenterHead, self).__init__(**kwargs)
        self.out_size_factor = self.train_cfg.get("out_size_factor")

        num_classes = [len(t["class_names"]) for t in kwargs["tasks"]]
        share_conv_channel = kwargs.get("share_conv_channel", 64)

        self.task_heads = nn.ModuleList()
        for idx, num_class in enumerate(num_classes):
            separate_head = detection_heads[idx].pop("separate_head")
            separate_head.update(in_channels=share_conv_channel, heads=detection_heads[idx], num_cls=num_class)
            self.task_heads.append(MODELS.build(separate_head))

    def multihead_targets_single(self, gt_instances_3d: InstanceData) -> Tuple[List[Tensor]]:
        """Generate training targets for a batch of samples."""

        gt_labels_3d = gt_instances_3d.labels_3d
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat((gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1).to(device)
        max_objs = self.train_cfg["max_objs"] * self.train_cfg["dense_reg"]
        grid_size = torch.tensor(self.train_cfg["grid_size"]).to(device)
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([torch.where(gt_labels_3d == class_name.index(i) + flag) for i in class_name])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            out_size_factor = self.out_size_factor[idx]
            feature_map_size = grid_size[:2] // out_size_factor
            heatmap = gt_bboxes_3d.new_zeros((len(self.class_names[idx]), feature_map_size[1], feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10), dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                length = task_boxes[idx][k][3]
                width = task_boxes[idx][k][4]
                length = length / voxel_size[0] / out_size_factor
                width = width / voxel_size[1] / out_size_factor

                if width > 0 and length > 0:
                    radius = gaussian_radius((width, length), min_overlap=self.train_cfg["gaussian_overlap"])
                    radius = max(self.train_cfg["min_radius"], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][1], task_boxes[idx][k][2]

                    coor_x = (x - pc_range[0]) / voxel_size[0] / out_size_factor
                    coor_y = (y - pc_range[1]) / voxel_size[1] / out_size_factor

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0] and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert y * feature_map_size[0] + x < feature_map_size[0] * feature_map_size[1]

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat(
                        [
                            center - torch.tensor([x, y], device=device),
                            z.unsqueeze(0),
                            box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            vx.unsqueeze(0),
                            vy.unsqueeze(0),
                        ]
                    )

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_instances_3d: InstanceData) -> Tuple[Tensor]:
        """Generate training targets for a single sample.

        Args:
            gt_instances_3d (:obj:`InstanceData`): Gt_instances of
                single data sample. It usually includes
                ``bboxes_3d`` and ``labels_3d`` attributes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """
        if not isinstance(self.out_size_factor, list):
            return super().get_targets_single(gt_instances_3d)
        else:
            return self.multihead_targets_single(gt_instances_3d)
