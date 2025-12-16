_base_ = [
    "bevfusion_lidar_voxel_second_secfpn_4xb8_120m_offline.py",
    "./models/bevfusion_camera_second_secfpn.py",
    "./pipelines/default_camera_lidar_120m.py",
    "./schedulers/default_30e_linear_cosine.py"
]

model = dict(
    type="BEVFusion",
    fusion_layer=dict(type="ConvFuser", in_channels=[80, 256], out_channels=256, kernel_size=5, padding=2),
    img_bev_bbox_head=_base_.img_bev_bbox_head,
)
