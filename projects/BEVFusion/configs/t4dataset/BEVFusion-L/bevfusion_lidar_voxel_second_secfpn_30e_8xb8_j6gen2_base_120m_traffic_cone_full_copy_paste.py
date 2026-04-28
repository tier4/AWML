_base_ = [
    "../../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/j6gen2_base.py",
    "../default/pipelines/default_lidar_intensity_120m.py",
    "../default/models/default_lidar_second_secfpn_120m.py",
    "../default/schedulers/default_30e_8xb8_adamw_cosine.py",
    "../default/default_misc.py",
]

custom_imports = dict(imports=["projects.BEVFusion.bevfusion"], allow_failed_imports=False)
custom_imports["imports"] += _base_.custom_imports["imports"]
custom_imports["imports"] += ["autoware_ml.detection3d.datasets.transforms"]

# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/kokseang_2_8/"

experiment_group_name = "bevfusion_lidar_intensity_traffic_cone/j6gen2_base/" + _base_.dataset_type
experiment_name = "lidar_voxel_second_secfpn_30e_8xb8_j6gen2_base_120m_traffic_cone_full_copy_paste"
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
        num_aug_features=5,
        # min-max normalization for x, y, z, intensity, time_lag, where the max of time lag technically is two seeps (200 ms) here
        aug_features_min_values=[
            _base_.point_cloud_range[0],
            _base_.point_cloud_range[1],
            _base_.point_cloud_range[2],
            0.0,
            0.0,
        ],
        aug_features_max_values=[
            _base_.point_cloud_range[3],
            _base_.point_cloud_range[4],
            _base_.point_cloud_range[5],
            255.0,
            0.2,
        ],
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
        )
    ),
)

db_sampler = dict(
    data_root=data_root,
    info_path=info_directory_path + _base_.info_train_file_name,
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            traffic_cone=5,
            barrier=5,
            bicycle=5,
            pedestrian=5)),
    classes=_base_.class_names,
    sample_groups=dict(
        car=0,
        truck=0,
        bus=0,
        barrier=2,
        traffic_cone=4),
    points_loader=dict(
        type='LoadPointsFromCurrentFileSweep',
        coord_type='LIDAR',
        load_dim=_base_.point_load_dim,
        use_dim=_base_.point_use_dim,
        backend_args=_base_.backend_args,
        sweeps_num=_base_.sweeps_num,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=False,
    ))
        
train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=_base_.point_load_dim,
        use_dim=_base_.point_use_dim,
        backend_args=_base_.backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=_base_.sweeps_num,
        load_dim=_base_.point_load_dim,
        use_dim=_base_.point_use_dim,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=_base_.backend_args,
        test_mode=False,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type="ObjectSample", db_sampler=db_sampler),
    dict(
        type="BEVFusionGlobalRotScaleTrans",
        scale_ratio_range=[0.95, 1.05],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=[0.5, 0.5, 0.2],
    ),
    dict(type="BEVFusionRandomFlip3D"),
    dict(type="PointsRangeFilter", point_cloud_range=_base_.point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=_base_.point_cloud_range),
    dict(
        type="ObjectNameFilter",
        classes=[
            "car",
            "truck",
            "bus",
            "bicycle",
            "pedestrian",
            "traffic_cone",
            "barrier",
        ],
    ),
    dict(type="PointShuffle"),
    dict(
        type="Pack3DDetInputs",
        keys=["points", "img", "gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "gt_labels"],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "img_path",
            "transformation_3d_flow",
            "pcd_rotation",
            "pcd_scale_factor",
            "pcd_trans",
            "img_aug_matrix",
            "lidar_aug_matrix",
            "timestamp",
            "vehicle_type",
            "city",
            "traffic_cone_barrier_status",
        ],
    ),
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
        test_mode=True,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="Pack3DDetInputs",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "img_path",
            "num_pts_feats",
            "num_views",
            "timestamp",
            "vehicle_type",
            "city",
            "traffic_cone_barrier_status",  
        ],
    ),
]

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

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="NuScenes metric/T4Metric/mAP"),
)
log_processor = dict(window_size=50)

load_from = "work_dirs/bevfusion_lidar_2.7.0/base/epoch_48.pth"

custom_hooks = []
