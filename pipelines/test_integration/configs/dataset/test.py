_base_ = ["../../../../autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py"]

custom_imports = dict(
    imports=[
        "autoware_ml.detection3d.datasets.t4dataset",
        "autoware_ml.detection3d.evaluation.t4metric",
    ]
)

dataset_version_config_root = "pipelines/test_integration/configs/dataset"
dataset_version_list = [
    "db_jpntaxi_v1",
    "db_jpntaxi_v2",
    "db_jpntaxi_v3",
    "db_gsm8_v1",
    "db_j6_v1",
]
