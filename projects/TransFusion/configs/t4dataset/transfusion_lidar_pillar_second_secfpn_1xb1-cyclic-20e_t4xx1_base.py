_base_ = [
    "../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py",
]
custom_imports = dict(
    imports=["projects.TransFusion.transfusion"], allow_failed_imports=False)
custom_imports["imports"] += _base_.custom_imports["imports"]

# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/user_name/"
val_interval = 5
max_epochs = 50
backend_args = None

# range setting (This si tentative parameter, and should be override)
point_cloud_range = [-12.8, -12.8, -3.0, 12.8, 12.8, 7.0]
voxel_size = [0.05, 0.05, 10]
grid_size = [512, 512, 1]

# model parameter
out_size_factor = 4
max_voxels = (1000, 1000)
pillar_feat_channels = [32]
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type="TransFusion",
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=max_voxels,
        ),
    ),
    pts_voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=5,
        feat_channels=pillar_feat_channels,
        voxel_size=voxel_size,
        norm_cfg=dict(type="BN1d", eps=0.001, momentum=0.01),
        point_cloud_range=point_cloud_range,
        with_distance=False,
    ),
    pts_middle_encoder=dict(
        type="PointPillarsScatter", in_channels=64, output_shape=grid_size),
    pts_backbone=dict(
        type="SECOND",
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    pts_bbox_head=dict(
        type="TransFusionHead",
        num_proposals=500,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=_base_.num_class,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type="TransformerDecoderLayer",
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            norm_cfg=dict(type="LN"),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
        ),
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type="TransFusionBBoxCoder",
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction="mean",
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type="mmdet.L1Loss", reduction="mean", loss_weight=0.25),
        loss_heatmap=dict(
            type="mmdet.GaussianFocalLoss", reduction="mean", loss_weight=1.0),
    ),
    train_cfg=dict(
        val_interval=val_interval,
        pts=dict(
            dataset="nuScenes",
            assigner=dict(
                type="HungarianAssigner3D",
                iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
                cls_cost=dict(
                    type="mmdet.FocalLossCost",
                    gamma=2,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
                iou_cost=dict(type="IoU3DCost", weight=0.25),
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range,
        )),
    test_cfg=dict(
        pts=dict(
            dataset="nuScenes",
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )),
)

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CBGSDataset",
        dataset=dict(
            data_root=data_root,
            ann_file=info_directory_path + _base_.info_train_file_name,
            modality=input_modality,
            type=_base_.dataset_type,
            metainfo=_base_.metainfo,
            class_names=_base_.class_names,
            test_mode=False,
            data_prefix=_base_.data_prefix,
            box_type_3d="LiDAR",
            backend_args=backend_args,
        ),
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_test_file_name,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    backend_args=backend_args,
    metric="bbox",
    class_names=_base_.class_names,
    data_mapping=_base_.name_mapping,
)
test_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_test_file_name,
    backend_args=backend_args,
    metric="bbox",
    class_names=_base_.class_names,
    data_mapping=_base_.name_mapping,
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend")
]
visualizer = dict(
    type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")

# learning rate
lr = 0.0001
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        T_max=20,
        eta_min=lr * 10,
        begin=0,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=30,
        eta_min=lr * 1e-4,
        begin=20,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=20,
        eta_min=0.85 / 0.95,
        begin=0,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=30,
        eta_min=1,
        begin=20,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(
    by_epoch=True, max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically or not by default.
auto_scale_lr = dict(enable=False)
log_processor = dict(window_size=50)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1))
custom_hooks = [dict(type="DisableObjectSampleHook", disable_after_epoch=40)]
