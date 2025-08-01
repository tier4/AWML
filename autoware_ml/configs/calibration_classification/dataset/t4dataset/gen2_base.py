custom_imports = dict(
    imports=[
        "autoware_ml.calibration_classification.datasets.t4_calibration_classification_dataset",
        "autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform",
        "autoware_ml.calibration_classification.hooks.result_visualization_hook",
    ],
    allow_failed_imports=False,
)

# dataset type setting
dataset_type = "T4Dataset"
info_train_file_name = "t4dataset_gen2_base_infos_train.pkl"
info_val_file_name = "t4dataset_gen2_base_infos_val.pkl"
info_test_file_name = "t4dataset_gen2_base_infos_test.pkl"


# dataset scene setting
dataset_version_config_root = "autoware_ml/configs/t4dataset"
dataset_version_list = [
    "db_j6gen2_v1",
    "db_j6gen2_v2",
    "db_j6gen2_v3",
    "db_j6gen2_v4",
]
