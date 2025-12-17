_base_ = [
    "../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/j6gen2_base.py",
    "../default/bevfusion_lidar_intensity_voxel_second_secfpn_4xb8_120m_offline.py",
]

# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/user_name/"

# Scheduler parameters
max_epochs = 30

model = dict(
    bbox_head=dict(
        class_names=_base_.class_names,  # Use class names to identify the correct class indices
    ),
)

train_dataloader = dict(
    dataset=dict(
        type=_base_.dataset_type,
        pipeline=_base_.train_pipeline,
        modality=_base_.input_modality,
        backend_args=_base_.backend_args,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_train_file_name,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        test_mode=False,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
    )
)

val_dataloader = dict(
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=_base_.test_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=_base_.input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=_base_.backend_args,
    )
)

test_dataloader = dict(
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_test_file_name,
        pipeline=_base_.test_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=_base_.input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=_base_.backend_args,
    )
)

val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    filter_attributes=_base_.filter_attributes,
)

test_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_test_file_name,
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    filter_attributes=_base_.filter_attributes,
)


# Redefine scheduler
param_scheduler = [
    # learning rate scheduler
    # During the first (max_epochs * 0.3) epochs, learning rate increases from 0 to lr * 10
    # during the next epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        T_max=8,
        eta_min=_base_.lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=22,
        eta_min=_base_.lr * 1e-4,
        begin=8,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first (0.3 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=22,
        eta_min=1,
        begin=8,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# runtime settings
# Run validation for every val_interval epochs before max_epochs - 10, and run validation every 2 epoch after max_epochs - 10
train_cfg = dict(
    by_epoch=True, max_epochs=max_epochs, val_interval=_base_.val_interval, dynamic_intervals=[(max_epochs - 5, 1)]
)

val_cfg = dict()
test_cfg = dict()

load_from = "<checkpoint>"
