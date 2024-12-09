_base_ = ["../../../../autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py"]

custom_imports = dict(
    imports=[
        "autoware_ml.detection3d.datasets.t4dataset",
        "autoware_ml.detection3d.evaluation.t4metric",
    ]
)

dataset_version_config_root = "pipelines/test_integration/configs/dataset"
dataset_version_list = [
    "database_v1_0",
    "database_v1_1",
    "database_v1_3",
    # https://github.com/tier4/autoware-ml/issues/278
    # "database_v2_0",
    "database_v3_0",
]
