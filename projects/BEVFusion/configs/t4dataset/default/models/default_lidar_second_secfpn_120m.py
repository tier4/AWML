num_proposals = 500
max_num_points = 32
max_voxels = [120000, 160000]
out_size_factor = 8

model = dict(
    type="BEVFusion",
    voxelize_cfg=dict(
        max_num_points=max_num_points,
        max_voxels=max_voxels,
    ),
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        pad_size_divisor=32,
    ),
    pts_voxel_encoder=dict(
        type="HardSimpleVoxelSinCosEncoder",
        in_channels=4,
    ),
    pts_middle_encoder=dict(
        type="BEVFusionSparseEncoder",
        in_channels=5,
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
            dataset="t4datasets",
            out_size_factor=out_size_factor,
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
            out_size_factor=out_size_factor,
            nms_type="circle",  # Set to "circle" for circle_nms
            # Set NMS for different clusters
            nms_clusters=[
                # Sqrt(0.25) = 0.50
                dict(
                    class_names=["car", "truck", "bus"], class_indices=[0, 1, 2], nms_threshold=0.25, post_max_size=300
                ),  # It's radius if using circle_nms
                dict(class_names=["bicycle"], class_indices=[3], nms_threshold=0.0, post_max_size=50),
                dict(class_names=["pedestrian"], class_indices=[4], nms_threshold=0.0, post_max_size=100),
                dict(class_names=["traffic_cone"], class_indices=[5], nms_threshold=0.0, post_max_size=100),
                dict(class_names=["barrier"], class_indices=[6], nms_threshold=0.0, post_max_size=50),
            ],
        ),
        dense_heatmap_pooling_classes=["car", "truck", "bus", "barrier"],  # Use class indices for pooling
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type="TransFusionBBoxCoder",
            post_center_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
            # score_threshold=0.03,
            # CAR, TRUCK, BUS, BICYCLE, PEDESTRIAN, TRAFFIC_CONE, BARRIER
            score_threshold=[0.015, 0.010, 0.010, 0.020, 0.030, 0.040, 0.020],
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
        loss_iou=None,
        loss_heatmap=dict(type="mmdet.GaussianFocalLoss", reduction="none", loss_weight=1.0),
        loss_bbox=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=0.25),
        # partial_
        partial_ignore_labels=["traffic_cone", "barrier"],
    ),
)
