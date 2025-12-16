_base_ = [
    "bevfusion_lidar_voxel_second_secfpn_4xb8_120m.py",
    "./models/bevfusion_camera_second_secfpn.py",
    "./pipelines/default_camera_lidar_120m.py",
]

max_epochs = 30
val_interval = 5

model = dict(
    type="BEVFusion",
    fusion_layer=dict(type="ConvFuser", in_channels=[80, 256], out_channels=256, kernel_size=5, padding=2),
    img_bev_bbox_head=_base_.img_bev_bbox_head,
)

# learning rate
# lr = 0.0001
lr = 1e-4
t_max = 3
param_scheduler = [
    # learning rate scheduler
	dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=t_max, by_epoch=True),
    dict(
        type="CosineAnnealingLR",
        T_max=(max_epochs - t_max),
        eta_min=lr * 1e-4,
        begin=t_max,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first (0.4 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=t_max,
        eta_min=0.85 / 0.95,
        begin=0,
        end=t_max,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=(max_epochs - t_max),
        eta_min=1,
        begin=t_max,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(
    by_epoch=True, max_epochs=max_epochs, val_interval=val_interval, dynamic_intervals=[(max_epochs - 5, 2)]
)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)