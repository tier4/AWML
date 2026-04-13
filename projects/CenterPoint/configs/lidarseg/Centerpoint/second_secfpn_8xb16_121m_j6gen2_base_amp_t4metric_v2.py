_base_ = [
    "second_secfpn_8xb16_121m_j6gen2_base_amp.py",
]

custom_imports = dict(
    imports=_base_.custom_imports["imports"] + ["autoware_ml.detection3d.datasets.transforms"],
    allow_failed_imports=False,
)

experiment_name = "second_secfpn_8xb16_121m_j6gen2_base_amp_t4metric_v2"
work_dir = "work_dirs/" + _base_.experiment_group_name + "/" + experiment_name

# Override filenames to use lidarseg specific ones as requested
info_test_file_name = "t4dataset_j6gen2_lidarseg_infos_test.pkl"
info_val_file_name = "t4dataset_j6gen2_lidarseg_infos_val.pkl"

info_train_statistics_file_name = "t4dataset_j6gen2_lidarseg_statistics_train.parquet"
info_val_statistics_file_name = "t4dataset_j6gen2_lidarseg_statistics_val.parquet"
info_test_statistics_file_name = "t4dataset_j6gen2_lidarseg_statistics_test.parquet"

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=_base_.point_load_dim,
        use_dim=_base_.point_load_dim,
        backend_args=_base_.backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=_base_.sweeps_num,
        load_dim=_base_.point_load_dim,
        use_dim=_base_.lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=_base_.backend_args,
        test_mode=True,
    ),
    dict(
        type="PointOffset",
        offset=[0.0, 0.0, 0.0],  # [dx, dy, dz] - adjust as needed
        sensor_id="LIDAR_FRONT_UPPER",  # Target specific LiDAR by name (newly supported)
        sensor_dim=4,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=_base_.point_cloud_range),
    dict(
        type="Pack3DDetInputs",
        keys=["points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=(
            "timestamp",
            "lidar2img",
            "depth2img",
            "cam2img",
            "box_type_3d",
            "sample_idx",
            "sample_token",
            "lidar_path",
            "ori_cam2img",
            "cam2global",
            "lidar2cam",
            "ego2global",
            "city",
            "vehicle_type",
            "lidar_sources_info",  # Required for range-based PointOffset
            "lidar_sources",  # Required for name-based PointOffset
        ),
    ),
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        ann_file=_base_.info_directory_path + info_val_file_name,
    )
)

test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        ann_file=_base_.info_directory_path + info_test_file_name,
    )
)

# Add evaluator configs
perception_evaluator_configs = dict(
    dataset_paths=_base_.data_root,
    frame_id="base_link",
    evaluation_config_dict=_base_.evaluator_metric_configs,
    load_raw_data=False,
)

frame_pass_fail_config = dict(
    target_labels=_base_.class_names,
    # Matching thresholds per class (must align with `plane_distance_thresholds` used in evaluation)
    matching_threshold_list=[2.0, 2.0, 2.0, 2.0, 2.0],
    confidence_threshold_list=None,
)

training_statistics_parquet_path = _base_.data_root + _base_.info_directory_path + info_train_statistics_file_name
testing_statistics_parquet_path = _base_.data_root + _base_.info_directory_path + info_test_statistics_file_name
validation_statistics_parquet_path = _base_.data_root + _base_.info_directory_path + info_val_statistics_file_name

val_evaluator = dict(
    _delete_=True,
    type="T4MetricV2",
    data_root=_base_.data_root,
    ann_file=_base_.data_root + _base_.info_directory_path + info_val_file_name,
    training_statistics_parquet_path=training_statistics_parquet_path,
    testing_statistics_parquet_path=testing_statistics_parquet_path,
    validation_statistics_parquet_path=validation_statistics_parquet_path,
    output_dir="validation",
    dataset_name="j6gen2_base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=None,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=8,
    scene_batch_size=-1,
    write_metric_summary=False,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    experiment_name=experiment_name,
    experiment_group_name=_base_.experiment_group_name,
)

test_evaluator = dict(
    _delete_=True,
    type="T4MetricV2",
    data_root=_base_.data_root,
    ann_file=_base_.data_root + _base_.info_directory_path + info_test_file_name,
    training_statistics_parquet_path=training_statistics_parquet_path,
    testing_statistics_parquet_path=testing_statistics_parquet_path,
    validation_statistics_parquet_path=validation_statistics_parquet_path,
    output_dir="testing",
    dataset_name="j6gen2_base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=None,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=8,
    scene_batch_size=-1,
    write_metric_summary=True,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    experiment_name=experiment_name,
    experiment_group_name=_base_.experiment_group_name,
)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="T4MetricV2/T4MetricV2/mAP_center_distance_bev"
    ),
)
