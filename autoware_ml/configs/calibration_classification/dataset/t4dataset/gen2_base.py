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


transform_config = dict(
    crop_ratio=0.6,  # Ratio for image cropping (0.0-1.0, where 1.0 means no crop)
    max_distortion=0.001,  # Maximum affine distortion as fraction of image dimensions (0.0-1.0)
    depth_scale=80.0,  # Depth scaling factor in meters (used to normalize depth values to 0-255 range)
    radius=2,  # Radius in pixels for LiDAR point visualization (creates 5x5 patches around each point)
    rgb_weight=0.3,  # Weight for RGB component in overlay visualization (0.0-1.0, where 1.0 is full RGB)
)
