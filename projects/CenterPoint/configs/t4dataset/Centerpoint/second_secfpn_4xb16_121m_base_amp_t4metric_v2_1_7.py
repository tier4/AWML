_base_ = [
    "./second_secfpn_4xb16_121m_base_amp_1_7.py",
]

work_dir = "work_dirs/centerpoint/" + _base_.dataset_type + "/centerpoint_2_5/second_secfpn_4xb16_121m_base_amp_t4metric_v2/"

# Add evaluator configs
perception_evaluator_configs = dict(
    dataset_paths=_base_.data_root,
    frame_id="base_link",
    result_root_directory=work_dir + "/result",
    evaluation_config_dict=_base_.evaluator_metric_configs,
    load_raw_data=False,
)

critical_object_filter_config = dict(
    target_labels=_base_.class_names,
    ignore_attributes=None,
    max_distance_list=[121.0, 121.0, 121.0, 121.0, 121.0],
    min_distance_list=[-121.0, -121.0, -121.0, -121.0, -121.0],
)

frame_pass_fail_config = dict(
    target_labels=_base_.class_names,
    # Matching thresholds per class (must align with `plane_distance_thresholds` used in evaluation)
    matching_threshold_list=[2.0, 2.0, 2.0, 2.0, 2.0],
    confidence_threshold_list=None,
)

val_evaluator = dict(
    _delete_=True,
    type="T4MetricV2",
    data_root=_base_.data_root,
    ann_file=_base_.data_root + _base_.info_directory_path + _base_.info_val_file_name,
    output_dir="validation",
    dataset_name="base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=critical_object_filter_config,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=64,
    scene_batch_size=-1,
    write_metric_summary=False,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
)

test_evaluator = dict(
    _delete_=True,
    type="T4MetricV2",
    data_root=_base_.data_root,
    ann_file=_base_.data_root + _base_.info_directory_path + _base_.info_test_file_name,
    output_dir="testing",
    dataset_name="base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=critical_object_filter_config,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=64,
    scene_batch_size=-1,
    write_metric_summary=True,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
)
