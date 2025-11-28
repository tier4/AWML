# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import CameraInstance3DBoxes, Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.structures.ops.transforms import bbox3d2result
from mmengine.runner.amp import autocast
from mmengine.structures import InstanceData

from projects.StreamPETR.stream_petr.models.utils.grid_mask import GridMask
from projects.StreamPETR.stream_petr.models.utils.misc import locations


@MODELS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    INPUT_TENSORS = [
        "lidar2img",
        "intrinsics",
        "extrinsics",
        "timestamp",
        "img_timestamp",
        "ego_pose",
        "ego_pose_inv",
        "img",
        "img_feats",
        "prev_exists",
    ]

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        num_frame_head_grads=2,
        num_frame_backbone_grads=2,
        num_frame_losses=2,
        stride=16,
        position_level=0,
        aux_2d_only=True,
        **kwargs,
    ):
        super(Petr3D, self).__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False

    def extract_img_feat(self, img, len_queue=1):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN / B / len_queue), C, H, W)

        return img_feats_reshaped

    def extract_feat(self, img, T):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T)
        return img_feats

    def obtain_history_memory(
        self,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_bboxes=None,
        gt_bboxes_labels=None,
        img_metas=None,
        centers_2d=None,
        depths=None,
        **data,
    ):
        losses = dict()
        T = data["img"].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in self.INPUT_TENSORS:
                data_t[key] = data[key][:, i]
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(
                gt_bboxes_3d[i],
                gt_labels_3d[i],
                gt_bboxes[i],
                gt_bboxes_labels[i],
                img_metas[i],
                centers_2d[i],
                depths[i],
                requires_grad=requires_grad,
                return_losses=return_losses,
                **data_t,
            )
            if loss is not None:
                for key, value in loss.items():
                    losses["frame_" + str(i) + "_" + key] = value
        return losses

    def prepare_location(self, shape, **data):
        pad_h, pad_w, _ = shape
        bs, n = data["img_feats"].shape[:2]
        x = data["img_feats"].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs * n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {"topk_indexes": None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi

    def forward_pts_train(
        self,
        gt_bboxes_3d,
        gt_labels_3d,
        gt_bboxes,
        gt_bboxes_labels,
        img_metas,
        centers_2d,
        depths,
        requires_grad=True,
        return_losses=False,
        **data,
    ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_location([x[0] for x in img_metas["pad_shape"]], **data)

        if not requires_grad:
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_metas, None, gt_bboxes_3d, gt_labels_3d, **data)
        else:
            outs_roi = self.forward_roi_head(location, **data)
            topk_indexes = outs_roi["topk_indexes"]
            outs = self.pts_bbox_head(location, img_metas, topk_indexes, gt_bboxes_3d, gt_labels_3d, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_bboxes_labels, centers_2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d)

            return losses
        else:
            return None

    def stack_tensors(self, data: dict):
        for key, values in data.items():
            if isinstance(values[0], torch.Tensor):
                # Stack the tensors along the first dimension
                data[key] = torch.stack(values, dim=0)

    def forward(self, mode="loss", **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        self.stack_tensors(data)
        if mode == "loss":
            return self.forward_train(**data)
        elif mode == "predict":
            with autocast(dtype=None, cache_enabled=False):
                return self.forward_test(**data)
        else:
            raise NotImplementedError()

    def forward_train(
        self,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_bboxes_labels=None,
        gt_bboxes=None,
        gt_bboxes_ignore=None,
        depths=None,
        centers_2d=None,
        **data,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_bboxes_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag:  # for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        T = data["img"].size(1)
        prev_img = data["img"][:, : -self.num_frame_backbone_grads]
        rec_img = data["img"][:, -self.num_frame_backbone_grads :]
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        if T - self.num_frame_backbone_grads > 0:
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T - self.num_frame_backbone_grads)
            data["img_feats"] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        else:
            data["img_feats"] = rec_img_feats
        losses = self.obtain_history_memory(
            gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_bboxes_labels, img_metas, centers_2d, depths, **data
        )
        return losses

    def forward_test(self, img_metas, **data):
        assert len(img_metas) == 1, "Test should be done in streaming manner, only batch size 1 is supported currently"
        return self.simple_test(img_metas, **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        location = self.prepare_location([x[0] for x in img_metas["pad_shape"]], **data)
        outs_roi = self.forward_roi_head(location, **data)
        topk_indexes = outs_roi["topk_indexes"]

        # Confirm if this logic works for t4 dataset
        if img_metas["scene_token"][0] and img_metas["scene_token"] != self.prev_scene_token:
            print("Test: Resetting memory due to new scene token")
            self.prev_scene_token = img_metas["scene_token"]
            data["prev_exists"] = data["img"].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data["prev_exists"] = data["prev_exists"]
        outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(outs)

        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results

    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data["img_feats"] = self.extract_img_feat(data["img"], 1)

        data_t = dict()
        for key in self.INPUT_TENSORS:
            data_t[key] = data[key][:, 0]
        results_3d = self.simple_test_pts(img_metas[0], **data_t)

        predictions = []
        for res_3d in results_3d:
            pred_instances_3d = InstanceData()
            pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(tensor=res_3d["bboxes_3d"], box_dim=9)
            pred_instances_3d.scores_3d = res_3d["scores_3d"]
            pred_instances_3d.labels_3d = res_3d["labels_3d"]
            predictions.append(
                Det3DDataSample(
                    pred_instances_3d=pred_instances_3d,
                    pred_instances=InstanceData(),
                    sample_idx=img_metas[0]["sample_idx"][0],
                )
            )

        return predictions

    def train(self, mode: bool = True):
        if self.training != mode:
            self.pts_bbox_head.reset_memory()
            print("Cleared memory due to change in mode. Train mode: ", mode)
            super().train(mode)
