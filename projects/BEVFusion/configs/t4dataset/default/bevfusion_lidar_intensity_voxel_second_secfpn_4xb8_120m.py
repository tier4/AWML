_base_ = [
    "bevfusion_lidar_voxel_second_secfpn_4xb8_120m.py",
    "./pipelines/default_lidar_intensity_120m.py",
]

model = dict(
    pts_voxel_encoder=dict(num_features=_base_.point_use_dim),
    pts_middle_encoder=dict(in_channels=_base_.point_use_dim),
)
