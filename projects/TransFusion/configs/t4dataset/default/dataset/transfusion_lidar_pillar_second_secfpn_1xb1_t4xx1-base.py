_base_ = [
    "../../../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py",
]
custom_imports = dict(
    imports=["projects.TransFusion.transfusion"],
    allow_failed_imports=False,
)
custom_imports["imports"] += _base_.custom_imports["imports"]

# user setting
info_directory_path = "info/user_name/"
data_root = "data/t4dataset/"
max_epochs = 50
backend_args = None
lr = 0.0001  # learning rate

# dataset parameter
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
)

# dataset
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CBGSDataset",
        dataset=dict(
            data_root=data_root,
            ann_file=info_directory_path + _base_.info_train_file_name,
            modality=input_modality,
            type=_base_.dataset_type,
            metainfo=_base_.metainfo,
            class_names=_base_.class_names,
            test_mode=False,
            data_prefix=_base_.data_prefix,
            box_type_3d="LiDAR",
            backend_args=backend_args,
        ),
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_test_file_name,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    backend_args=backend_args,
    metric="bbox",
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
)
test_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_test_file_name,
    backend_args=backend_args,
    metric="bbox",
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(
    type="Det3DLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically or not by default.
auto_scale_lr = dict(enable=False)
log_processor = dict(window_size=50)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3,save_best='NuScenes metric/T4Metric/mAP'),
)
custom_hooks = [dict(type="DisableObjectSampleHook", disable_after_epoch=40)]

param_scheduler = [
    # learning rate scheduler
    # During the first (max_epochs * 0.4) epochs, learning rate increases from 0 to lr * 10
    # during the next epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        T_max=int(max_epochs * 0.4),
        eta_min=lr * 10,
        begin=0,
        end=int(max_epochs * 0.4),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=int(max_epochs * 0.6),
        eta_min=lr * 1e-4,
        begin=int(max_epochs * 0.4),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first (0.4 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=int(max_epochs * 0.4),
        eta_min=0.85 / 0.95,
        begin=0,
        end=int(max_epochs * 0.4),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=int(max_epochs * 0.6),
        eta_min=1,
        begin=int(max_epochs * 0.4),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# Default setting for scaling LR automatically `base_batch_size` = (1 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(base_batch_size=1)
