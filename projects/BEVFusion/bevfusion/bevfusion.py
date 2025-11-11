from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from .ops import Voxelization


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        pts_backbone: dict,
        pts_neck: dict,
        bbox_head: dict,
        voxelize_cfg: Optional[dict] = None,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        img_roi_head = None,
        img_aux_bbox_head = None,
        img_aux_bbox_head_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if voxelize_cfg is not None:
            self.voxelize_reduce = voxelize_cfg.pop("voxelize_reduce")
            self.pts_voxel_layer = Voxelization(**voxelize_cfg)
            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
            self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        else:
            self.pts_voxel_layer = None
            self.voxelize_reduce = False
            self.pts_voxel_encoder = None
            self.pts_middle_encoder = None

        # Image Backbone, Neck and View Transformer
        if img_backbone is not None:
            assert img_neck is not None, "img_neck should be passed when img_backbone is passed"
            assert view_transform is not None, "view_transform should be passed when img_backbone is passed"

            self.img_backbone = MODELS.build(img_backbone)
            self.img_neck = MODELS.build(img_neck)
            self.view_transform = MODELS.build(view_transform)
        else:
            self.img_backbone = None
            self.img_neck = None
            self.view_transform = None
        
        if img_aux_bbox_head is not None:
            self.img_aux_bbox_head = MODELS.build(img_aux_bbox_head)
        else:
            self.img_aux_bbox_head = None 

        self.img_aux_bbox_head_weight = img_aux_bbox_head_weight
        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)
        else:
            self.fusion_layer = None

        # BEV Backbone and Neck
        if pts_backbone:
            self.pts_backbone = MODELS.build(pts_backbone)
        else:
            self.pts_backbone = None 
        
        if pts_neck:
            self.pts_neck = MODELS.build(pts_neck)
        else:
            self.pts_neck = None 

        if img_roi_head is not None:
            self.img_roi_head = MODELS.build(img_roi_head)

        self.bbox_head = MODELS.build(bbox_head)
        self.init_weights()

    def _forward(self, batch_inputs_dict: Tensor, batch_data_samples: OptSampleList = None, **kwargs):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """

        # NOTE(knzo25): this is used during onnx export
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head(feats, batch_input_metas)

        return outputs[0][0]

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(value for key, value in log_vars if "loss" in key)
        log_vars.insert(0, ["loss", loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head."""
        return hasattr(self, "seg_head") and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        camera_intrinsics_inverse=None,
        img_aug_matrix_inverse=None,
        lidar_aug_matrix_inverse=None,
        geom_feats=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if self.img_roi_head is not None:
            img_roi_head_preds = self.img_roi_head(x)
        else:
            img_roi_head_preds = None

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        assert BN == B * N, (BN, B * N)
        x = x.view(B, N, C, H, W)

        with torch.cuda.amp.autocast(enabled=False):
            # with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
                camera_intrinsics_inverse,
                img_aug_matrix_inverse,
                lidar_aug_matrix_inverse,
                geom_feats,
            )

        return x, img_roi_head_preds

    def extract_pts_feat(self, feats, coords, sizes, points=None) -> torch.Tensor:
        if points is not None:
            # NOTE(knzo25): training and normal inference
            with torch.cuda.amp.autocast(enabled=False):
                # with torch.autocast('cuda', enabled=False):
                points = [point.float() for point in points]
                feats, coords, sizes = self.voxelize(points)
                batch_size = coords[-1, 0] + 1
        else:
            # NOTE(knzo25): onnx inference. Voxelization happens outside the graph
            with torch.cuda.amp.autocast(enabled=False):
                # with torch.autocast('cuda', enabled=False):

                # NOTE(knzo25): onnx demmands this
                # batch_size = coords[-1, 0] + 1
                batch_size = 1
                print("Run onnx point_eSpConvst")
                assert self.voxelize_reduce
                if self.voxelize_reduce:
                    feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(
        self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs
    ) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, img_feats, img_roi_head_preds = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)
            # outputs = self.bbox_head.predict(feats, batch_data_samples)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get("imgs", None)
        points = batch_inputs_dict.get("points", None)
        img_feature = None
        features = []
        img_feature = None
        if imgs is not None and "lidar2img" not in batch_inputs_dict:
            # NOTE(knzo25): normal training and testing
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta["lidar2img"])
                camera_intrinsics.append(meta["cam2img"])
                camera2lidar.append(meta["cam2lidar"])
                img_aug_matrix.append(meta.get("img_aug_matrix", np.eye(4)))
                lidar_aug_matrix.append(meta.get("lidar_aug_matrix", np.eye(4)))

            lidar2image = imgs.new_tensor(np.array(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.array(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.array(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.array(lidar_aug_matrix))
            img_feature, img_roi_head_preds = self.extract_img_feat(
                x=imgs,
                points=deepcopy(points),
                lidar2image=lidar2image,
                camera_intrinsics=camera_intrinsics,
                camera2lidar=camera2lidar,
                img_aug_matrix=img_aug_matrix,
                lidar_aug_matrix=lidar_aug_matrix,
                img_metas=batch_input_metas,
            )
            features.append(img_feature)
        elif imgs is not None:
            # NOTE(knzo25): onnx inference
            lidar2image = batch_inputs_dict["lidar2img"]
            camera_intrinsics = batch_inputs_dict["cam2img"]
            camera2lidar = batch_inputs_dict["cam2lidar"]
            img_aug_matrix = batch_inputs_dict["img_aug_matrix"]
            lidar_aug_matrix = batch_inputs_dict["lidar_aug_matrix"]

            # NOTE(knzo25): originally BEVFusion uses all the points
            # which could be a bit slow. For now I am using only
            # the centroids, which is also suboptimal, but using
            # all the voxels produce errors in TensorRT,
            # so this will be fixed for the next version
            # (ScatterElements bug, or simply null voxels break the equation)
            feats = batch_inputs_dict["voxels"]["voxels"]
            sizes = batch_inputs_dict["voxels"]["num_points_per_voxel"]

            feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)

            geom_feats = batch_inputs_dict["geom_feats"]
            img_feature, img_roi_head_preds = self.extract_img_feat(
                imgs,
                [feats],
                # points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
                geom_feats=geom_feats,
            )
            features.append(img_feature)

        if points is not None and self.pts_middle_encoder is not None:
            pts_feature = self.extract_pts_feat(
                batch_inputs_dict.get("voxels", {}).get("voxels", None),
                batch_inputs_dict.get("voxels", {}).get("coors", None),
                batch_inputs_dict.get("voxels", {}).get("num_points_per_voxel", None),
                points=points,
            )
            features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        if self.pts_backbone:
            x = self.pts_backbone(x)
        
        if self.pts_neck:
            x = self.pts_neck(x)

        return x, img_feature, img_roi_head_preds

    def loss(
        self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs
    ) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, img_feats, img_roi_head_preds = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
        
        losses.update(bbox_loss)
        
        if self.img_aux_bbox_head:
            img_aux_bbox_losses = self.img_aux_bbox_head.loss([img_feats], batch_data_samples)
            sum_losses = 0.0
            weighted_sum_losses = 0.0
            for loss_key, loss in img_aux_bbox_losses.items():
                sum_losses += loss
                losses[loss_key] = loss * self.img_aux_bbox_head_weight
                weighted_sum_losses += losses[loss_key]

            losses["img_aux_sum"] = sum_losses
            losses["img_aux_weighted_sum"] = weighted_sum_losses

            # losses.update(img_aux_bbox_loss)

        if self.img_roi_head is not None:
            gt_bboxes2d_list = []
            gt_labels2d_list = []
            centers2d = [] 
            depths = []
            img_pad_shapes = []
            gt_bboxes_ignore = None 

            for batch_data_sample in batch_data_samples:
                gt_bboxes2d_list.append(
                    batch_data_sample.gt_bboxes
                )
                gt_labels2d_list.append(
                    batch_data_sample.gt_labels2d
                )
                centers2d.append(
                    batch_data_sample.center2d
                )
                depths.append(
                    batch_data_sample.depths
                )
                if gt_bboxes_ignore in batch_data_sample:
                    batch_input_metas.append(
                        gt_bboxes_ignore
                    ) 
            
            for batch_input_meta in batch_input_metas:
                img_pad_shapes.append(
                    batch_input_meta.pad_shape
                )

            img_roi_head_losses = self.img_roi_head.aux_loss(
                gt_bboxes2d_list=gt_bboxes2d_list,
                gt_labels2d_list=gt_labels2d_list,
                centers2d=centers2d, 
                depths=depths, 
                preds_dicts=img_roi_head_preds,
                img_pad_shapes=img_pad_shapes
                gt_bboxes_ignore=gt_bboxes_ignore
            )

            losses.update(
                img_roi_head_losses
            )


        return losses
