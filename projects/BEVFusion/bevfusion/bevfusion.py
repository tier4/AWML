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
        data_preprocessor: OptConfigType = None,
        voxelize_cfg: Optional[dict] = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        loss_depth_weight: float = 3.0,
        depth_gt_downsample: int = 1,
        **kwargs,
    ) -> None:
        """Initialize BEVFusion model.

        Args:
            data_preprocessor (dict): Data preprocessor config.
            voxelize_cfg (dict): Voxelization config.
            pts_voxel_encoder (dict): Point voxel encoder config.
            pts_middle_encoder (dict): Point middle encoder config.
            fusion_layer (dict): Fusion layer config.
            img_backbone (dict): Image backbone config.
            img_neck (dict): Image neck config.
            pts_backbone (dict): Point backbone config.
            pts_neck (dict): Point neck config.
            bbox_head (dict): Bbox head config.
            init_cfg (dict): Initialization config.
            seg_head (dict): Segmentation head config.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if voxelize_cfg is not None:
            self.pts_voxel_layer = Voxelization(**voxelize_cfg)
            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
            self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        else:
            self.pts_voxel_layer = None
            self.pts_voxel_encoder = None
            self.pts_middle_encoder = None

        self.img_backbone = MODELS.build(img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(view_transform) if view_transform is not None else None

        self.fusion_layer = MODELS.build(fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone) if pts_backbone is not None else None
        self.pts_neck = MODELS.build(pts_neck) if pts_neck is not None else None

        self.bbox_head = MODELS.build(bbox_head)
        self._weights_initialized = False
        self.loss_depth_weight = loss_depth_weight
        self.depth_gt_downsample = depth_gt_downsample

    def _forward(
        self, batch_inputs_dict: Tensor, batch_data_samples: OptSampleList = [], using_image_features=False, **kwargs
    ):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """

        # NOTE(knzo25): this is used during onnx export
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas, using_image_features)

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
        if self._weights_initialized:
            return
        if self.img_backbone is not None:
            self.img_backbone.init_weights()
        self._weights_initialized = True

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head."""
        return hasattr(self, "seg_head") and self.seg_head is not None
    
    def prepare_camera_depth_aware_parameters(
        self, 
        camera_intrinsics: torch.Tensor, 
        img_aug_matrix: torch.Tensor,
        lidar_aug_matrix: torch.Tensor,
        camera2lidar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            camera_intrinsics: torch.Tensor, the camera intrinsics of shape (B, N, 3, 3).
            img_aug_matrix: torch.Tensor, the image augmentation matrix of shape (B, N, 4, 4).
            lidar_aug_matrix: torch.Tensor, the lidar augmentation matrix of shape (B, 4, 4).
            camera2lidar: torch.Tensor, the camera to lidar matrix of shape (B, N, 4, 4).
        Returns:
            torch.Tensor, the camera depth aware parameters of shape (B*N, N_CAMERA_DEPTH_PARAMETERS).
        """
        B, N, _, _ = camera_intrinsics.shape
        lidar_aug_matrix = lidar_aug_matrix.view(B, 1, 4, 4).repeat(1, N, 1, 1)
        
        # (B*N, 15)
        mlp_input = torch.stack([
            camera_intrinsics[:, :, 0, 0],   # fx
            camera_intrinsics[:, :, 1, 1],   # fy
            camera_intrinsics[:, :, 0, 2],   # cx
            camera_intrinsics[:, :, 1, 2],   # cy
            img_aug_matrix[:, :, 0, 0],   # r11
            img_aug_matrix[:, :, 0, 1],   # r12
            img_aug_matrix[:, :, 0, 3],   # t1
            img_aug_matrix[:, :, 1, 0],   # r21
            img_aug_matrix[:, :, 1, 1],   # r22
            img_aug_matrix[:, :, 1, 3],   # t2
            lidar_aug_matrix[:, :, 0, 0],   # r11
            lidar_aug_matrix[:, :, 0, 1],   # r12
            lidar_aug_matrix[:, :, 1, 0],   # r21
            lidar_aug_matrix[:, :, 1, 1],   # r22
            lidar_aug_matrix[:, :, 2, 2],   # r33
        ], dim=-1)
        # (B, N, 4, 4) -> (B, N, 3, 4) -> (B*N, 12)
        camera2lidar_flatten = camera2lidar[:,:,:3,:].view(B, N, -1)
        
        # (B, N, 15+12)
        mlp_input = torch.cat([mlp_input, camera2lidar_flatten], dim=-1)
        return mlp_input

    def get_image_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        assert BN == B * N, (BN, B * N)
        x = x.view(B, N, C, H, W)
        return x

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
        using_image_features=False,
        camera_depth_aware_parameters=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not using_image_features:
            x = self.get_image_backbone_features(x)

        with torch.amp.autocast("cuda",enabled=False):
            # with torch.autocast(device_type='cuda', dtype=torch.float32):
            x, pred_depths = self.view_transform(
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
                camera_depth_aware_parameters=camera_depth_aware_parameters
            )
        return x, pred_depths

    def extract_pts_feat(self, feats, coords, sizes, points=None) -> torch.Tensor:
        if points is not None:
            # NOTE(knzo25): training and normal inference
            with torch.amp.autocast("cuda", enabled=False):
                # with torch.autocast('cuda', enabled=False):
                points = [point.float() for point in points]
                feats, coords, sizes = self.voxelize(points)
                batch_size = coords[-1, 0] + 1
        else:
            # NOTE: (knzo25): onnx inference. Voxelization happens outside the graph
            with torch.amp.autocast("cuda", enabled=False):
                # with torch.autocast('cuda', enabled=False):

                # NOTE(knzo25): onnx demmands this
                # batch_size = coords[-1, 0] + 1
                # with torch.autocast('cuda', enabled=False):

                # NOTE(knzo25): onnx demmands this
                # batch_size = coords[-1, 0] + 1
                batch_size = 1
                print("Run onnx point_eSpConvst")
        feats = self.pts_voxel_encoder(feats, sizes, coords)
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
        assert len(sizes) > 0, "No points in the voxel"
        sizes = torch.cat(sizes, dim=0)
        
        # if self.voxelize_reduce:
        #     feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
        #     feats = feats.contiguous()
        
        return feats, coords, sizes

    def predict(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
        using_image_features=False,
        **kwargs,
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
        feats, _ = self.extract_feat(batch_inputs_dict, batch_input_metas, using_image_features)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        using_image_features,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get("imgs", None)
        points = batch_inputs_dict.get("points", None)
        features = []

        is_onnx_inference = False
        pred_depths = None
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

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            camera_depth_aware_parameters = self.prepare_camera_depth_aware_parameters(
                camera_intrinsics=camera_intrinsics,
                img_aug_matrix=img_aug_matrix,
                lidar_aug_matrix=lidar_aug_matrix,
                camera2lidar=camera2lidar,
            )
            img_feature, pred_depths = self.extract_img_feat(
                imgs,
                deepcopy(points),
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
                using_image_features=using_image_features,
                camera_depth_aware_parameters=camera_depth_aware_parameters,
            )
            features.append(img_feature)
        elif imgs is not None:
            # NOTE(knzo25): onnx inference
            is_onnx_inference = True
            lidar2image = batch_inputs_dict["lidar2img"]
            camera_intrinsics = batch_inputs_dict["cam2img"]
            camera2lidar = batch_inputs_dict["cam2lidar"]
            img_aug_matrix = batch_inputs_dict["img_aug_matrix"]
            lidar_aug_matrix = batch_inputs_dict["lidar_aug_matrix"]
            geom_feats = batch_inputs_dict["geom_feats"]
            # Retrieve the parameters from deployment code directly
            camera_depth_aware_parameters = batch_inputs_dict["camera_depth_aware_parameters"]

            img_feature, pred_depths = self.extract_img_feat(
                imgs,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
                geom_feats=geom_feats,
                using_image_features=using_image_features,
                camera_depth_aware_parameters=camera_depth_aware_parameters,
            )
            features.append(img_feature)

        if self.pts_middle_encoder is not None:
            pts_feature = self.extract_pts_feat(
                batch_inputs_dict.get("voxels", {}).get("voxels", None),
                batch_inputs_dict.get("voxels", {}).get("coors", None),
                batch_inputs_dict.get("voxels", {}).get("num_points_per_voxel", None),
                points=points if not is_onnx_inference else None,
            )
            features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        if self.pts_backbone is not None:
            x = self.pts_backbone(x)

        if self.pts_neck is not None:
            x = self.pts_neck(x)

        return x, pred_depths

    def loss(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
        using_image_features: bool = False,
        **kwargs,
    ) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, pred_depths = self.extract_feat(batch_inputs_dict, batch_input_metas, using_image_features)

        losses = dict()
        if self.loss_depth_weight > 0 and pred_depths is not None:
            with torch.amp.autocast("cuda", enabled=False):
                gt_depths = torch.stack(
                    [
                        meta["gt_depths"]
                        if isinstance(meta["gt_depths"], torch.Tensor)
                        else torch.as_tensor(meta["gt_depths"])
                        for meta in batch_input_metas
                    ]
                ).to(device=pred_depths.device, dtype=torch.float32)
                depth_loss = self.get_depth_loss(gt_depths, pred_depths)
                losses["loss_depth"] = depth_loss
        
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
            losses.update(bbox_loss)

        return losses
 
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        D = self.view_transform.D
        dbounds = self.view_transform.dbound
        gt_depths = gt_depths.view(B * N, H // self.depth_gt_downsample,
                                   self.depth_gt_downsample, W // self.depth_gt_downsample,
                                   self.depth_gt_downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.depth_gt_downsample * self.depth_gt_downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.depth_gt_downsample,
                                   W // self.depth_gt_downsample)

        gt_depths = (gt_depths - (dbounds[0] - dbounds[2])) / dbounds[2]
        # gt_depths = torch.where(gt_depths >= 0.0, gt_depths, torch.zeros_like(gt_depths))
        # gt_depths = torch.clamp(gt_depths, max=float(D))
        gt_depths = torch.where((gt_depths >= 0.0) & (gt_depths < D + 1), gt_depths, torch.zeros_like(gt_depths))
        # gt_depths = torch.clamp(gt_depths, max=float(D))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=D + 1).view(-1, D + 1)[:, 1:]
        return gt_depths.float()

    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # (B, N, D, H, W) -> (B*N*H*W, D)
        depth_preds = depth_preds.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.view_transform.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        depth_loss = F.binary_cross_entropy(
            depth_preds,
            depth_labels,
            reduction='none',
        ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss