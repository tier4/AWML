_base_ = ["../../../../../autoware_ml/configs/detection3d/default_runtime.py"]

# range setting
point_cloud_range = [-14.4, -14.4, -3.0, 14.4, 14.4, 5.0]
voxel_size = [0.01, 0.01, 0.01]
grid_size = [1440, 1440, 41]
eval_class_range = {
    "car": 121,
    "truck": 121,
    "bus": 121,
    "bicycle": 121,
    "pedestrian": 121,
}

# model parameter
input_modality = dict(use_lidar=True, use_camera=False)
point_load_dim = 5  # x, y, z, intensity, ring_id
point_use_dim = 5
point_intensity_dim = 3
max_num_points = 10
max_voxels = [120000, 160000]
num_proposals = 500

model = dict(
    type="BEVFusion",
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=max_num_points,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=max_voxels,
            voxelize_reduce=True,
        ),
    ),
    pts_voxel_encoder=dict(type="HardSimpleVFE", num_features=5),
    pts_middle_encoder=dict(
        type="BEVFusionSparseEncoder",
        in_channels=5,
        aug_features=False,
        aug_features_min_values=[],
        aug_features_max_values=[],
        num_aug_features=0,
        sparse_shape=grid_size,
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
            dataset="nuScenes",
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
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
            dataset="nuScenes",
            grid_size=grid_size,
            out_size_factor=8,
            voxel_size=voxel_size[0:2],
            pc_range=point_cloud_range[0:2],
            nms_type=None,
        ),
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type="TransFusionBBoxCoder",
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[0:2],
            post_center_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
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

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="NuScenes metric/T4Metric/mAP"),
)
custom_hooks = [dict(type="DisableObjectSampleHook", disable_after_epoch=15)]
log_processor = dict(window_size=50)
