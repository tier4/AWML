_base_ = [
    "../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/base.py"
    "../default/bevfusion_camera_lidar_second_secfpn_4xb8_centerhead_aux_120m.py",
]

# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/user_name/"

model = dict(
    type="BEVFusion",
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
        filter_cfg=dict(filter_frames_with_camera_order=_base_.camera_order),
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
