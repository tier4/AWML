_base_ = [
	  "./bevfusion_camera_resnet50_fpn_lss_depthaware_30e_8xb32_j6gen2_base_50m.py",
]

experiment_name = "bevfusion_camera_resnet50_fpn_lss_v2_depthaware_30e_8xb32_j6gen2_base_50m_t4metric_v2"
work_dir = "work_dirs/" + _base_.experiment_group_name + "/" + experiment_name

# Add evaluator configs
evaluator_metric_configs = dict(
    evaluation_task="detection",
    target_labels=_base_.class_names,
    center_distance_bev_thresholds=[0.5, 1.0, 2.0, 4.0],
    # plane_distance_thresholds is required for the pass fail evaluation
    plane_distance_thresholds=[2.0, 4.0],
    iou_2d_thresholds=None,
    iou_3d_thresholds=None,
    label_prefix="autoware",
    # bev minimum distance ranges for each range bucket, must be the same length as max_distance,
    # they will form bev distance ranges in [(min_distance[0], max_distance[0]), (min_distance[1], max_distance[1]), ...] when filtering
    min_distance=[0.0],
    # bev maximum distance ranges for each range bucket, must be the same length as min_distance
    max_distance=[51.2],
		min_point_numbers=0,
    matching_class_agnostic_fps=False,
)

perception_evaluator_configs = dict(
    dataset_paths=_base_.data_root,
    frame_id="base_link",
    evaluation_config_dict=evaluator_metric_configs,
    load_raw_data=False,
)


frame_pass_fail_config = dict(
    target_labels=_base_.class_names,
    # Matching thresholds per class (must align with `plane_distance_thresholds` used in evaluation)
    matching_threshold_list=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    confidence_threshold_list=None,
)

training_statistics_parquet_path = (
    _base_.data_root + _base_.info_directory_path + _base_.info_train_statistics_file_name
)
testing_statistics_parquet_path = _base_.data_root + _base_.info_directory_path + _base_.info_test_statistics_file_name
validation_statistics_parquet_path = (
    _base_.data_root + _base_.info_directory_path + _base_.info_val_statistics_file_name
)

val_evaluator = dict(
    _delete_=True,
    type="T4MetricV2",
    data_root=_base_.data_root,
    ann_file=_base_.data_root + _base_.info_directory_path + _base_.info_val_file_name,
    training_statistics_parquet_path=training_statistics_parquet_path,
    testing_statistics_parquet_path=testing_statistics_parquet_path,
    validation_statistics_parquet_path=validation_statistics_parquet_path,
    output_dir="validation",
    dataset_name="j6gen2_base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=None,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=64,
    scene_batch_size=-1,
    write_metric_summary=False,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    experiment_name=experiment_name,
    experiment_group_name=_base_.experiment_group_name,
    min_num_points=2,
)

test_evaluator = dict(
    _delete_=True,
    type="T4MetricV2",
    data_root=_base_.data_root,
    ann_file=_base_.data_root + _base_.info_directory_path + _base_.info_test_file_name,
    training_statistics_parquet_path=training_statistics_parquet_path,
    testing_statistics_parquet_path=testing_statistics_parquet_path,
    validation_statistics_parquet_path=validation_statistics_parquet_path,
    output_dir="testing",
    dataset_name="j6gen2_base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=None,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=64,
    scene_batch_size=-1,
    write_metric_summary=True,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    experiment_name=experiment_name,
    experiment_group_name=_base_.experiment_group_name,
    min_num_points=2,
)
