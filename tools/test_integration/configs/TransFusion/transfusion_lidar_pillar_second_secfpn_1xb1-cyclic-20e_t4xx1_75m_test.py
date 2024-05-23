_base_ = [
    "../../../../projects/TransFusion/configs/t4dataset/transfusion_lidar_pillar_second_secfpn_1xb8-cyclic-20e_t4xx1_75m_precise.py",
]

train_gpu_size = 1
train_batch_size = 1
val_batch_size = 1
max_epochs = 1

data_root = "data/t4dataset/"
info_directory_path = "info/test_name/"
evaluation = dict(interval=1)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_batch_size,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CBGSDataset",
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            ann_file=info_directory_path + "t4dataset_xx1_infos_train.pkl",
            pipeline=_base_.train_pipeline,
            metainfo=_base_.metainfo,
            class_names=_base_.class_names,
            modality=_base_.input_modality,
            test_mode=False,
            data_prefix=_base_.data_prefix,
            box_type_3d="LiDAR",
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + "t4dataset_xx1_infos_val.pkl",
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
test_dataloader = val_dataloader

val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + "t4dataset_xx1_infos_val.pkl",
    metric="bbox",
    backend_args=_base_.backend_args,
    class_names=_base_.class_names,
    data_mapping=_base_.name_mapping,
    eval_class_range=_base_.eval_class_range,
)

test_evaluator = val_evaluator
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
auto_scale_lr = dict(
    enable=False, base_batch_size=train_gpu_size * train_batch_size)
