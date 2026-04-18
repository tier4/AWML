_base_ = [
    "./bevfusion_lidar_voxel_second_secfpn_30e_8xb8_j6gen2_base_120m.py",
]

# user setting
data_root = "data/t4dataset/"
info_directory_path = "info/user_name/"

experiment_group_name = "bevfusion_lidar_intensity_2.6.1/j6gen2_base/" + _base_.dataset_type
experiment_name = "lidar_voxel_second_secfpn_30e_8xb8_j6gen2_base_120m_t4metric_v2"
work_dir = "work_dirs/" + experiment_group_name + "/" + experiment_name

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
)
