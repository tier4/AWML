_base_ = [
    "../../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/base.py",
    "../default/pipelines/default_lidar_120m.py",
    "../models/default_lidar_second_secfpn_120m.py",
    "../default/schedulers/default_50e_4xb8_adamw_cosine.py",
    "../default/default_misc.py",
]

custom_imports = dict(imports=["projects.BEVFusion.bevfusion"], allow_failed_imports=False)
custom_imports["imports"] += _base_.custom_imports["imports"]

# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/user_name/"

experiment_group_name = "bevfusion_lidar/base/" + _base_.dataset_type
experiment_name = "lidar_voxel_second_secfpn_50e_4xb8_base_120m"
work_dir = "work_dirs/" + experiment_group_name + "/" + experiment_name

# model parameter
model = dict(
    type="BEVFusion",
    voxelize_cfg=dict(
        point_cloud_range=_base_.point_cloud_range,
        voxel_size=_base_.voxel_size,
        voxelize_reduce=True,
    ),
    pts_voxel_encoder=dict(num_features=_base_.point_use_dim),
    pts_middle_encoder=dict(
        in_channels=_base_.point_use_dim,
        sparse_shape=_base_.grid_size,
    ),
    bbox_head=dict(
        class_names=_base_.class_names,  # Use class names to identify the correct class indices
        train_cfg=dict(
            point_cloud_range=_base_.point_cloud_range,
            grid_size=_base_.grid_size,
            voxel_size=_base_.voxel_size,
        ),
        test_cfg=dict(
            grid_size=_base_.grid_size,
            voxel_size=_base_.voxel_size[0:2],
            pc_range=_base_.point_cloud_range[0:2],
        ),
        bbox_coder=dict(
            pc_range=_base_.point_cloud_range[0:2],
            voxel_size=_base_.voxel_size[0:2],
        ),
    ),
)

# Dataset parameters
train_dataloader = dict(
    batch_size=_base_.train_batch_size,
    num_workers=_base_.num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
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
        filter_cfg=_base_.filter_cfg,
    ),
)

val_dataloader = dict(
    batch_size=_base_.test_batch_size,
    num_workers=_base_.num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
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
    ),
)

test_dataloader = dict(
    batch_size=_base_.test_batch_size,
    num_workers=_base_.num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
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
    ),
)

val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    metric="bbox",
    backend_args=_base_.backend_args,
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=_base_.eval_class_range,
    filter_attributes=_base_.filter_attributes,
)

test_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_test_file_name,
    metric="bbox",
    backend_args=_base_.backend_args,
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=_base_.eval_class_range,
    filter_attributes=_base_.filter_attributes,
    save_csv=True,
)
