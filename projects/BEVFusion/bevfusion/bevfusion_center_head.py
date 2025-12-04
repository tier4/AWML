from typing import Tuple

import torch
from torch import Tensor
from mmdet3d.models.utils import clip_sigmoid, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.registry import MODELS
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from projects.CenterPoint.models.dense_heads.centerpoint_head import CenterHead


@MODELS.register_module(force=True)
class BEVFusionCenterHead(CenterHead):

    def __init__(
        self,
        freeze_shared_conv: bool = False,
        freeze_task_heads: bool = False,
        loss_prefix: str = "bevfusion_img",
        **kwargs,
    ):
        super(BEVFusionCenterHead, self).__init__(
            freeze_shared_conv=freeze_shared_conv,
            freeze_task_heads=freeze_task_heads,
            loss_prefix=loss_prefix,
            **kwargs,
        )

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
        gt_labels_3d = gt_instances_3d.labels_3d
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat((gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1).to(device)
        max_objs = self.train_cfg["max_objs"] * self.train_cfg["dense_reg"]
        grid_size = torch.tensor(self.train_cfg["grid_size"]).to(device)
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])

        feature_map_size = grid_size[:2] // self.train_cfg["out_size_factor"]

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
            heatmap = gt_bboxes_3d.new_zeros((len(self.class_names[idx]), feature_map_size[1], feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10), dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                length = task_boxes[idx][k][3]
                width = task_boxes[idx][k][4]
                length = length / voxel_size[0] / self.train_cfg["out_size_factor"]
                width = width / voxel_size[1] / self.train_cfg["out_size_factor"]

                if width > 0 and length > 0:
                    radius = gaussian_radius((width, length), min_overlap=self.train_cfg["gaussian_overlap"])
                    radius = max(self.train_cfg["min_radius"], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][1], task_boxes[idx][k][2]

                    dx, dy, dz = task_boxes[idx][k][3], task_boxes[idx][k][4], task_boxes[idx][k][5]
                    # Move to surface depth
                    # x = x - dx/2

                    coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg["out_size_factor"]
                    coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg["out_size_factor"]

                    center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0] and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int[[1, 0]], radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert y * feature_map_size[0] + x < feature_map_size[0] * feature_map_size[1]

                    ind[new_idx] = y * feature_map_size[0] + x
                    print(f"x: {x}, y: {y}, ind: {ind[new_idx]}")
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
