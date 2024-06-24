custom_imports = dict(imports=[
    "autoware_ml.detection3d.datasets.t4dataset",
    "autoware_ml.detection3d.evaluation.t4metric.t4metric",
])

# dataset type setting
dataset_type = "T4Dataset"
info_train_file_name = "t4dataset_x2_infos_train.pkl"
info_val_file_name = "t4dataset_x2_infos_val.pkl"
info_test_file_name = "t4dataset_x2_infos_test.pkl"

# dataset scene setting
dataset_version_config_root = "autoware_ml/configs/detection3d/dataset/t4dataset/"
dataset_version_list = ["database_v2_0", "database_v3_0"]

# dataset format setting
data_prefix = dict(pts="", sweeps="")
camera_types = {
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK",
    "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
}

# class setting
name_mapping = {
    "animal": "animal",
    "movable_object.barrier": "barrier",
    "movable_object.pushable_pullable": "pushable_pullable",
    "movable_object.traffic_cone": "traffic_cone",
    "pedestrian.adult": "pedestrian",
    "pedestrian.child": "pedestrian",
    "pedestrian.construction_worker": "pedestrian",
    "pedestrian.personal_mobility": "pedestrian",
    "pedestrian.police_officer": "pedestrian",
    "pedestrian.stroller": "pedestrian",
    "pedestrian.wheelchair": "pedestrian",
    "static_object.bicycle rack": "bicycle rack",
    "static_object.bollard": "bollard",
    "vehicle.ambulance": "truck",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "truck",
    "vehicle.fire": "truck",
    "vehicle.motorcycle": "bicycle",
    "vehicle.police": "car",
    "vehicle.trailer": "truck",
    "vehicle.truck": "truck",
}

class_names = [
    "car",
    "truck",
    "bus",
    "bicycle",
    "pedestrian",
]
num_class = len(class_names)
metainfo = dict(classes=class_names)
