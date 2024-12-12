# range parameter
point_cloud_range = [-122.88, -122.88, -3.0, 122.88, 122.88, 7.0]
voxel_size = [0.32, 0.32, 10]
grid_size = [768, 768, 1]

# model parameter
out_size_factor = 4
pillar_feat_channels = [64, 64]
max_voxels = (60000, 60000)

##############################
####### set parameters #######
##############################

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
        type="PointPillarsScatter",
        in_channels=64,
        output_shape=(grid_size[0], grid_size[1]),
    ),
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
            center=(2, 2),
            height=(1, 2),
            dim=(3, 2),
            rot=(2, 2),
            vel=(2, 2),
        ),
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
            type="mmdet.L1Loss",
            reduction="mean",
            loss_weight=0.25,
        ),
        loss_heatmap=dict(
            type="mmdet.GaussianFocalLoss",
            reduction="mean",
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(
        pts=dict(
            dataset="nuScenes",
            assigner=dict(
                type="HungarianAssigner3D",
                iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
                cls_cost=dict(
                    type="mmdet.FocalLossCost",
                    gamma=2,
                    alpha=0.25,
                    weight=0.15,
                ),
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
        )
    ),
    test_cfg=dict(
        pts=dict(
            dataset="nuScenes",
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )
    ),
)
