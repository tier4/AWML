# https://github.com/tianweiy/CenterPoint/blob/e14421df7b9fe86ae42b7349aeffed4705a9ab93/configs/centerpoint/nusc_centerpoint_pp_02voxel_circle_nms.py
# v1 of arxiv
out_size_factor = 1

# https://github.com/open-mmlab/mmdetection3d/blob/main/configs/_base_/models/centerpoint_pillar02_second_secfpn_nus.py
model = dict(
    type="CenterPoint",
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        voxel=True,
        voxel_layer=dict(max_num_points=20, voxel_size=[0.2, 0.2, 8], max_voxels=(32000, 60000)),
    ),
    pts_voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        legacy=False,
    ),
    pts_middle_encoder=dict(
        type="PointPillarsScatter",
        in_channels=64,
        output_shape=(512, 512),
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
        type="CenterHead",
        in_channels=sum([128, 128, 128]),
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
            out_size_factor=out_size_factor,
            code_size=9,
        ),
        share_conv_channel=64,
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=1),
        loss_cls=dict(type="mmdet.GaussianFocalLoss", reduction="mean", loss_weight=1.0),
        loss_bbox=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            # (Reg x 2, height x 1, dim 3, rot x 2, vel x 2)
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        )
    ),
    test_cfg=dict(
        pts=dict(
            nms_type="circle",
            min_radius=[1.0],
            post_max_size=100,
            # nms_type="rotate",
            # post_center_limit_range=[-90.0, -90.0, -10.0, 90.0, 90.0, 10.0],
            # score_threshold=0.1,
            # nms_thr=0.2,
            # pre_max_size=1000,
            # post_max_size=100,
        )
    ),
)
