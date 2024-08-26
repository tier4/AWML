_base_ = [
    "../../../../autoware_ml/configs/detection3d/dataset/t4dataset/xx1.py"
]

custom_imports = dict(imports=[
    "autoware_ml.detection3d.datasets.t4dataset",
    "autoware_ml.detection3d.evaluation.t4metric",
])

dataset_version_config_root = "pipelines/test_integration/configs/dataset"
dataset_version_list = ["database_v1_1"]
