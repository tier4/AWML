custom_imports = dict(imports=[
    "autoware_ml.detection3d.datasets.t4dataset",
    "autoware_ml.detection3d.evaluation.t4metric",
])

dataset_version_config_root = "tools/test_integration/configs/dataset"
dataset_version_list = ["database_v1_1"]

camera_types = {
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK",
    "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
}

name_mapping = {
    # DBv1.0
    "vehicle.car": "car",
    "vehicle.construction": "truck",
    "vehicle.emergency (ambulance & police)": "car",
    "vehicle.motorcycle": "bicycle",
    "vehicle.trailer": "truck",
    "vehicle.truck": "truck",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus (bendy & rigid)": "bus",
    "pedestrian.adult": "pedestrian",
    "pedestrian.child": "pedestrian",
    "pedestrian.construction_worker": "pedestrian",
    "pedestrian.personal_mobility": "pedestrian",
    "pedestrian.police_officer": "pedestrian",
    "pedestrian.stroller": "pedestrian",
    "pedestrian.wheelchair": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "debris",
    "movable_object.pushable_pullable": "pushable_pullable",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.traffic_cone": "traffic_cone",
    "animal": "animal",
    "static_object.bicycle_rack": "bicycle_rack",
    # DBv1.1 and UCv2.0
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "trailer": "truck",
    "motorcycle": "bicycle",
    "bicycle": "bicycle",
    "police_car": "car",
    "pedestrian": "pedestrian",
    "police_officer": "pedestrian",
    "forklift": "car",
    "construction_worker": "pedestrian",
    "stroller": "pedestrian",
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
