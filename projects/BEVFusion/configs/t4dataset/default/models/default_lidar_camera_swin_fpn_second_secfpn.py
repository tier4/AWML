_base_ = [
    "./default_lidar_second_secfpn_120m.py",
]

# Image network
model = dict(
    voxelize_cfg=None,
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        pad_size_divisor=32,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        rgb_to_bgr=False,
    ),
    pts_voxel_encoder=None,
    pts_middle_encoder=None,
    img_backbone=dict(
        type="mmdet.SwinTransformer",
        pretrain_img_size=(256, 704),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type="Pretrained",
            # https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth
            checkpoint="work_dirs/swin_transformer/swint_nuimages_pretrained.pth",  # noqa: E251
        ),
    ),
    img_neck=dict(
        type="GeneralizedLSSFPN",
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        act_cfg=dict(type="ReLU", inplace=True),
        upsample_cfg=dict(mode="bilinear", align_corners=False),
    ),
    view_transform=dict(
        type="DepthLSSTransform",
        in_channels=256,
        out_channels=80,
        feature_size=[48, 96],
        xbound=[-122.40, 122.40, 0.68],
        ybound=[-122.40, 122.40, 0.68],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 130, 1.0],
        downsample=2,
    ),
)
