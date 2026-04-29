_base_ = [
    "bevfusion_offline_lidar_voxel_convnext_secfpn_30e_4xb4_j6gen2_full_categories_120m.py",
]

# Dry-run: 3 epochs with OneCycleLR to verify convergence before full training.
# Batch halved (2 per GPU) to reduce peak memory; validate every epoch.
# If OOM persists, halve again (batch=1) or disable gradient checkpointing stress-test.
#
# Usage:
#   python tools/train.py <this_config> [--resume auto]
#   --resume auto restores model+optimizer from the latest checkpoint in work_dir.

experiment_name = "convnext_dryrun_4xb2"
work_dir = "work_dirs/" + _base_.experiment_group_name + "/" + experiment_name

train_batch_size = 2
max_epochs = 3
val_interval = 1
lr = 7.0e-5
num_workers = 2  # keep shm usage low: 2 workers × 4 ranks = 8 total

train_dataloader = dict(batch_size=train_batch_size, num_workers=num_workers, pin_memory=False)
val_dataloader = dict(num_workers=num_workers, pin_memory=False)
test_dataloader = dict(num_workers=num_workers, pin_memory=False)

auto_scale_lr = dict(enable=False, base_batch_size=4 * train_batch_size)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# OneCycleLR: aggressive warm-up then cosine decay — visible loss change in 2-3 epochs.
# total_steps is set to max_epochs; by_epoch=True means the scheduler steps once per epoch.
# pct_start=0.3 → 30% of epochs in warm-up phase (1 epoch here).
param_scheduler = [
    dict(
        type="OneCycleLR",
        eta_max=lr,
        total_steps=max_epochs,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,      # initial_lr = eta_max / div_factor
        final_div_factor=1e4, # min_lr = initial_lr / final_div_factor
        by_epoch=True,
        convert_to_iter_based=False,
    )
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=val_interval)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=20),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3,
                    save_best="NuScenes metric/T4Metric/mAP"),
)
log_processor = dict(window_size=20)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
    dict(
        type="MLflowVisBackend",
        save_dir="mlruns",
        exp_name=_base_.experiment_group_name,
        run_name=experiment_name,
        tags=dict(
            backbone="ConvNeXt_PC",
            dataset="j6gen2_full_categories",
            num_classes=str(len(_base_.class_names)),
            voxel_size="0.075_0.075_0.2",
            run_type="dryrun",
        ),
    ),
]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")
