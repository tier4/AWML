_base_ = [
    "../../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/base.py",
    "../../default/second_secfpn_base.py",
]
custom_imports = dict(imports=["projects.CenterPoint.models"], allow_failed_imports=False)
custom_imports["imports"] += _base_.custom_imports["imports"]
custom_imports["imports"] += ["autoware_ml.detection3d.datasets.transforms"]
custom_imports["imports"] += ["autoware_ml.hooks"]

# This is a base file for t4dataset, add the dataset config.
# type, data_root and ann_file of data.train, data.val and data.test
point_cloud_range = [-121.60, -121.60, -3.0, 121.60, 121.60, 5.0]
voxel_size = [0.32, 0.32, 8.0]
grid_size = [760, 760, 1]  # (121.60 / 0.32 == 380, 380 * 2 == 760)
sweeps_num = 1
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
)
out_size_factor = 1

backend_args = None
# backend_args = dict(backend="disk")
point_load_dim = 5  # x, y, z, intensity, ring_id
point_use_dim = 3  # x, y, z
lidar_sweep_dims = [0, 1, 2, 4]

# eval parameter
eval_class_range = {
    "car": 121,
    "truck": 121,
    "bus": 121,
    "bicycle": 121,
    "pedestrian": 121,
}

# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/user_name/"
train_gpu_size = 2
train_batch_size = 8
test_batch_size = 2
num_workers = 32
val_interval = 1
max_epochs = 50
work_dir = "work_dirs/centerpoint/" + _base_.dataset_type + "/second_secfpn_2xb8_121m_base_t4metric_v2/"

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=point_load_dim,
        use_dim=point_load_dim,
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=point_load_dim,
        use_dim=lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(
        type="RandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-1.571, 1.571],
        scale_ratio_range=[0.80, 1.20],
        translation_std=[1.0, 1.0, 0.2],
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes={{_base_.class_names}}),
    dict(type="ObjectMinPointsFilter", min_num_points=5),
    dict(type="PointShuffle"),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=point_load_dim,
        use_dim=point_load_dim,
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=point_load_dim,
        use_dim=lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=point_load_dim,
        use_dim=point_load_dim,
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=point_load_dim,
        use_dim=lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=_base_.dataset_type,
        pipeline=train_pipeline,
        modality=input_modality,
        backend_args=backend_args,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_train_file_name,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        test_mode=False,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
    ),
)
val_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_val_file_name,
        pipeline=test_pipeline,
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
    batch_size=test_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=_base_.dataset_type,
        data_root=data_root,
        ann_file=info_directory_path + _base_.info_test_file_name,
        pipeline=test_pipeline,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        modality=input_modality,
        data_prefix=_base_.data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
# Add evaluator configs
perception_evaluator_configs = dict(
    dataset_paths=data_root,
    frame_id="base_link",
    result_root_directory=work_dir + "/result",
    evaluation_config_dict=_base_.evaluator_metric_configs,
    load_raw_data=False,
)

critical_object_filter_config = dict(
    target_labels=_base_.class_names,
    ignore_attributes=None,
    max_distance_list=[121.0, 121.0, 121.0, 121.0, 121.0],
    min_distance_list=[0.0, 0.0, 0.0, 0.0, 0.0],
)

frame_pass_fail_config = dict(
    target_labels=_base_.class_names,
    matching_threshold_list=None,
    confidence_threshold_list=None,
)

val_evaluator = dict(
    type="T4MetricV2",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=critical_object_filter_config,
    frame_pass_fail_config=frame_pass_fail_config,
    save_preds_and_gt_to_pickle=False,
    load_preds_and_gt_from_pickle=False,
)

test_evaluator = dict(
    type="T4MetricV2",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_test_file_name,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=critical_object_filter_config,
    frame_pass_fail_config=frame_pass_fail_config,
    save_preds_and_gt_to_pickle=True,
    load_preds_and_gt_from_pickle=False,
    results_pickle_path="/workspace/pickles",
)

model = dict(
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_voxels=(32000, 60000),
            deterministic=True,
        ),
    ),
    # Use BackwardPillarFeatureNet without computing voxel center for z-dimensionality
    pts_voxel_encoder=dict(
        type="BackwardPillarFeatureNet",
        in_channels=4,
        feat_channels=[32, 32],
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        legacy=False,
    ),
    pts_middle_encoder=dict(type="PointPillarsScatter", in_channels=32, output_shape=(grid_size[0], grid_size[1])),
    pts_backbone=dict(
        type="SECOND",
        in_channels=32,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    pts_bbox_head=dict(
        type="CenterHead",
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=5, class_names=["car", "truck", "bus", "bicycle", "pedestrian"]),
        ],
        bbox_coder=dict(
            voxel_size=voxel_size,
            pc_range=point_cloud_range,
            # No filter by range
            post_center_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
            out_size_factor=out_size_factor,
        ),
        loss_cls=dict(type="mmdet.GaussianFocalLoss", reduction="none", loss_weight=1.0),
        loss_bbox=dict(type="mmdet.L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=out_size_factor,
        ),
    ),
    test_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            # No filter by range
            post_center_limit_range=[-200.0, -200.0, -10.0, 200.0, 200.0, 10.0],
        ),
    ),
)

randomness = dict(seed=0, diff_rank_seed=False, deterministic=True)

# learning rate
# Since mmengine doesn't support OneCycleMomentum yet, we use CosineAnnealing from the default configs
lr = 0.0003
param_scheduler = [
    # learning rate scheduler
    # During the first (max_epochs * 0.3) epochs, learning rate increases from 0 to lr * 10
    # during the next epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        T_max=int(max_epochs * 0.3),
        eta_min=lr * 10,
        begin=0,
        end=int(max_epochs * 0.3),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epochs - int(max_epochs * 0.3),
        eta_min=lr * 1e-4,
        begin=int(max_epochs * 0.3),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first (0.3 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=int(max_epochs * 0.3),
        eta_min=0.85 / 0.95,
        begin=0,
        end=int(max_epochs * 0.3),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=max_epochs - int(max_epochs * 0.3),
        eta_min=1,
        begin=int(max_epochs * 0.3),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (2 GPUs) x (8 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)
auto_scale_lr = dict(enable=False, base_batch_size=train_gpu_size * train_batch_size)

# Only set if the number of train_gpu_size more than 1
if train_gpu_size > 1:
    sync_bn = "torch"

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")

logger_interval = 50
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=logger_interval),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="NuScenes metric/T4Metric/mAP"),
)

custom_hooks = [
    dict(type="MomentumInfoHook"),
]
