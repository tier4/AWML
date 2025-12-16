_base_ = [
    "../../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "./pipelines/default_lidar_120m.py"
]

custom_imports = dict(imports=["projects.BEVFusion.bevfusion", "projects.CenterPoint.models"], allow_failed_imports=False)
custom_imports["imports"] += _base_.custom_imports["imports"]
custom_imports["imports"] += ["autoware_ml.detection3d.datasets.transforms"]

# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/user_name/"
train_gpu_size = 4
train_batch_size = 8
test_batch_size = 2
val_interval = 5
max_epochs = 50
num_workers = 32

eval_class_range = {
    "car": 121.0,
    "truck": 121.0,
    "bus": 121.0,
    "bicycle": 121.0,
    "pedestrian": 121.0,
}

# model parameter
input_modality = dict(use_lidar=True, use_camera=False)
max_num_points = 10
max_voxels = [120000, 160000]
num_proposals = 500
out_size_factor = 8

# Aux Image BEV head, this will be used in the downstream camera network
img_bev_bbox_head = dict(
    type="BEVFusionCenterHead",
    in_channels=80,
    # (output_channel_size, num_conv_layers)
    common_heads=dict(
        reg=(2, 2),
        height=(1, 2),
        dim=(3, 2),
        rot=(2, 2),
        vel=(2, 2),
    ),
    bbox_coder=dict(
        type="CenterPointBBoxCoder",
        max_num=500,
        score_threshold=0.1,
        code_size=9,
        voxel_size=_base_.voxel_size,
        pc_range=_base_.point_cloud_range,
        post_center_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
        out_size_factor=out_size_factor,
    ),
    share_conv_channel=64,
    loss_cls=dict(type="mmdet.GaussianFocalLoss", reduction="none", loss_weight=1.0),
    loss_bbox=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=0.0), # Dont need to learn regression in this aux head
    norm_bbox=True,
    tasks=[
        dict(num_class=5, class_names=["car", "truck", "bus", "bicycle", "pedestrian"]),
    ],
    # sigmoid(-4.595) = 0.01 for initial small values
    separate_head=dict(type="CustomSeparateHead", init_bias=-4.595, final_kernel=1),
    train_cfg=dict(
        out_size_factor=out_size_factor,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        # (Reg x 2, height x 1, dim 3, rot x 2, vel x 2)
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        grid_size=_base_.grid_size,
        voxel_size=_base_.voxel_size,
        point_cloud_range=_base_.point_cloud_range,
    ),
    test_cfg=dict(
        nms_type="circle",
        min_radius=[1.0],
        post_max_size=100,
        grid_size=_base_.grid_size,
        out_size_factor=out_size_factor,
        pc_range=_base_.point_cloud_range,
        voxel_size=_base_.voxel_size,
        # No filter by range
        post_center_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
    ),
)

model = dict(
    type="BEVFusion",
    voxelize_cfg=dict(
    		max_num_points=max_num_points,
    		voxel_size=_base_.voxel_size,
    		point_cloud_range=_base_.point_cloud_range,
    		max_voxels=max_voxels,
    		deterministic=True,
    		voxelize_reduce=True,
    ),
    data_preprocessor=None,
    pts_voxel_encoder=dict(type="HardSimpleVFE", num_features=_base_.point_use_dim),
    pts_middle_encoder=dict(
        type="BEVFusionSparseEncoder",
        in_channels=_base_.point_use_dim,
        aug_features_min_values=[],
        aug_features_max_values=[],
        num_aug_features=0,
        sparse_shape=_base_.grid_size,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type="basicblock",
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    fusion_layer=None,
    bbox_head=dict(
        type="BEVFusionHead",
        num_proposals=num_proposals,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
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
        train_cfg=dict(
            dataset="t4datasets",
            point_cloud_range=_base_.point_cloud_range,
            grid_size=_base_.grid_size,
            voxel_size=_base_.voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type="HungarianAssigner3D",
                iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
                cls_cost=dict(type="mmdet.FocalLossCost", gamma=2.0, alpha=0.25, weight=0.15),
                reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
                iou_cost=dict(type="IoU3DCost", weight=0.25),
            ),
        ),
        test_cfg=dict(
            dataset="t4datasets",
            grid_size=_base_.grid_size,
            out_size_factor=out_size_factor,
            voxel_size=_base_.voxel_size[0:2],
            pc_range=_base_.point_cloud_range[0:2],
            nms_type=None,  # Set to "circle" for circle_nms
            # Set NMS for different clusters
            nms_clusters=[
                dict(class_names=["car", "truck", "bus"], nms_threshold=0.5),  # It's radius if using circle_nms
                dict(class_names=["bicycle"], nms_threshold=0.5),
                dict(class_names=["pedestrian"], nms_threshold=0.175),
            ],
        ),
        dense_heatmap_pooling_classes=["car", "truck", "bus", "bicycle"],  # Use class indices for pooling
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type="TransFusionBBoxCoder",
            pc_range=_base_.point_cloud_range[0:2],
            voxel_size=_base_.voxel_size[0:2],
            post_center_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
            score_threshold=0.0,
            out_size_factor=out_size_factor,
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
        loss_heatmap=dict(type="mmdet.GaussianFocalLoss", reduction="mean", loss_weight=1.0),
        loss_bbox=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=0.25),
    ),
)


train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=None    # Must be overridden by a sub-config
)

val_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=None    # Must be overriden by a sub-config
)

test_dataloader = dict(
    batch_size=_base_.test_batch_size,
    num_workers=_base_.num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=None    # Must be overriden by a sub-config
)

val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    metric="bbox",
    backend_args=_base_.backend_args,
    eval_class_range=eval_class_range,
    
    # Must be overriden by a sub-config
    ann_file=None,
    class_names=[],
    name_mapping=None,
    filter_attributes=None,
)

test_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    metric="bbox",
    backend_args=_base_.backend_args,
    eval_class_range=eval_class_range,
    save_csv=True,
    
    # Must be overriden by a sub-config
    ann_file=None,
    class_names=[],
    name_mapping=None,
    filter_attributes=None,
)

# learning rate
lr = 0.0001
t_max = 8
param_scheduler = [
    # learning rate scheduler
    # During the first (max_epochs * 0.4) epochs, learning rate increases from 0 to lr * 10
    # during the next epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        T_max=t_max,
        eta_min=lr * 10,
        begin=0,
        end=t_max,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=(max_epochs - t_max),
        eta_min=lr * 1e-4,
        begin=t_max,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first (0.4 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=t_max,
        eta_min=0.85 / 0.95,
        begin=0,
        end=t_max,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=(max_epochs - t_max),
        eta_min=1,
        begin=t_max,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# runtime settings
# Run validation for every val_interval epochs before max_epochs - 10, and run validation every 2 epoch after max_epochs - 10
train_cfg = dict(
    by_epoch=True, max_epochs=max_epochs, val_interval=val_interval, dynamic_intervals=[(max_epochs - 5, 2)]
)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)
auto_scale_lr = dict(enable=False, base_batch_size=train_gpu_size * train_batch_size)

# Only set if the number of train_gpu_size more than 1
if train_gpu_size > 1:
    sync_bn = "torch"

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="NuScenes metric/T4Metric/mAP"),
)
log_processor = dict(window_size=50)

randomness = dict(seed=0, diff_rank_seed=True, deterministic=True)
