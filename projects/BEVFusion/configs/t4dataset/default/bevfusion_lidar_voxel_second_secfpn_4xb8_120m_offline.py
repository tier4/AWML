_base_ = [
    "bevfusion_lidar_voxel_second_secfpn_4xb8_120m.py"
]

# range setting
voxel_size = [0.075, 0.075, 0.2]
grid_size = [3264, 3264, 41]

model = dict(
    type="BEVFusion",
    voxelize_cfg=dict(
        voxel_size=voxel_size,
    ),
    bbox_head=dict(
        train_cfg=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
        ),
        test_cfg=dict(
            grid_size=grid_size,
        ),
        bbox_coder=dict(
            voxel_size=voxel_size[0:2],
        ),
    ),
)

