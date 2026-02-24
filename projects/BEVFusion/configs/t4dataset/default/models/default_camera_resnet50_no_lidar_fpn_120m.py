_base_ = [
    "./default_lidar_second_secfpn_120m.py",
]

# Image network
model = dict(
	  data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        pad_size_divisor=32,
				# mean=[103.530, 116.280, 123.675], 
				# std=[57.375, 57.120, 58.395],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        rgb_to_bgr=False,
    ),
    # Remove all lidar related configs
    voxelize_cfg=None,
    pts_voxel_encoder=None,
    pts_middle_encoder=None,
    pts_neck=None,
    pts_backbone=None,
		img_backbone=dict(
        # pretrained="torchvision://resnet50",
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style="pytorch",
				init_cfg=dict(
            type="Pretrained",
            # https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth
            checkpoint="work_dirs/resnet50/mmdet_resnet50-19c8e357.pth",  # noqa: E251
        ),
				strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
    ),
    img_neck=dict(
        type="GeneralizedLSSFPN",
        in_channels=[1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        act_cfg=dict(type="ReLU", inplace=True),
        upsample_cfg=dict(mode="bilinear", align_corners=False),
    ),
    view_transform=dict(
        type="LSSTransform",
        in_channels=256,
        out_channels=80,
        feature_size=[24, 48],
        xbound=[-122.40, 122.40, 0.68],
        ybound=[-122.40, 122.40, 0.68],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 130, 1.0],
        downsample=2,
    ),
)
