_base_ = [
    "bevfusion_lidar_intensity_voxel_second_secfpn_4xb8_120m.py",
    "./models/bevfusion_camera_second_secfpn.py"
    "./pipelines/default_camera_lidar_intensity_120m.py",
    "./schedulers/default_30e_linear_cosine.py"
]

model = dict(
    type="BEVFusion",
    voxelize_cfg=None,
    pts_voxel_encoder=None,
    pts_middle_encoder=None,
    pts_backbone=dict(
        type="SECOND",
        in_channels=80,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    img_bev_bbox_head=_base_.img_bev_bbox_head
)
