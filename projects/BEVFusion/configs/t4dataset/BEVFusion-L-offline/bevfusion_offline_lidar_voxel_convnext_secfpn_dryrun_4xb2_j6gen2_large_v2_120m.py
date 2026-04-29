_base_ = [
    "./bevfusion_offline_lidar_voxel_convnext_secfpn_30e_4xb4_j6gen2_large_v2_120m.py",
]

# Smoke-test: 3 epochs, OneCycleLR, batch=2
# Purpose: verify convergence signal and catch crashes before 50-epoch run.
experiment_name = "convnext_v2_6sweep_large_dryrun"
work_dir = "work_dirs/" + _base_.experiment_group_name + "/" + experiment_name

train_batch_size = 2
max_epochs = 3
val_interval = 1

train_dataloader = dict(batch_size=train_batch_size)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=7.0e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(
        type="OneCycleLR",
        max_lr=7.0e-4,
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=1e4,
        by_epoch=False,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=val_interval)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
    dict(
        type="MLflowVisBackend",
        save_dir="mlruns",
        exp_name=_base_.experiment_group_name,
        run_name=experiment_name,
        tags=dict(
            backbone="ConvNeXt_PC_large_arch",
            dataset="j6gen2_large",
            num_classes="20",
            voxel_size="0.075_0.075_0.2",
            sweeps="6",
        ),
    ),
]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")
