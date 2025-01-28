custom_imports = dict(
    imports=[
        "autoware_ml.detection3d.datasets.t4dataset",
        "autoware_ml.detection3d.evaluation.t4metric.t4metric",
    ]
)

# dataset type setting
dataset_type = "T4Dataset"
info_train_file_name = "t4dataset_xx1_infos_train.pkl"
info_val_file_name = "t4dataset_xx1_infos_val.pkl"
info_test_file_name = "t4dataset_xx1_infos_test.pkl"

# dataset scene setting
dataset_version_config_root = "autoware_ml/configs/t4dataset/"
dataset_version_list = [
    "db_jpntaxi_v1",
    "db_jpntaxi_v2",
    "db_jpntaxi_v4",
]

# dataset format setting
data_prefix = dict(
    pts="",
    sweeps="",
    CAM_FRONT="",
    CAM_FRONT_LEFT="",
    CAM_FRONT_RIGHT="",
    CAM_BACK="",
    CAM_BACK_RIGHT="",
    CAM_BACK_LEFT="",
)
camera_types = {
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
}

# class setting
name_mapping = {
    # DBv1.0
    "vehicle.car": "car",
    "vehicle.construction": "truck",
    "vehicle.emergency (ambulance & police)": "car",
    "vehicle.motorcycle": "bicycle",
    "vehicle.trailer": "trailer",
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
    "trailer": "trailer",
    "motorcycle": "bicycle",
    "bicycle": "bicycle",
    "police_car": "car",
    "pedestrian": "pedestrian",
    "police_officer": "pedestrian",
    "forklift": "car",
    "construction_worker": "pedestrian",
    "stroller": "pedestrian",
    # DBv1.3
    "semi_trailer": "trailer",
    "tractor_unit": "truck",
    "kart": "car",
    "unknown": "unknown",
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

merge_objects = [
    ("truck", ["truck", "trailer"]),
]
merge_type = "extend_longer"  # One of ["extend_longer","union", None]

# visualization
class_colors = {
    "car": (30, 144, 255),
    "truck": (140, 0, 255),
    "construction_vehicle": (255, 255, 0),
    "bus": (111, 255, 111),
    "trailer": (0, 255, 255),
    "barrier": (0, 0, 0),
    "motorcycle": (100, 0, 30),
    "bicycle": (255, 0, 30),
    "pedestrian": (255, 200, 200),
    "traffic_cone": (120, 120, 120),
}
camera_panels = [
    "data/CAM_FRONT_LEFT",
    "data/CAM_FRONT",
    "data/CAM_FRONT_RIGHT",
    "data/CAM_BACK_LEFT",
    "data/CAM_BACK",
    "data/CAM_BACK_RIGHT",
]

# Add filter attributes
filter_attributes = [
    ("vehicle.bicycle", "vehicle_state.parked"),
    ("vehicle.bicycle", "cycle_state.without_rider"),
    ("vehicle.bicycle", "motorcycle_state.without_rider"),
    ("vehicle.motorcycle", "vehicle_state.parked"),
    ("vehicle.motorcycle", "cycle_state.without_rider"),
    ("vehicle.motorcycle", "motorcycle_state.without_rider"),
    ("bicycle", "vehicle_state.parked"),
    ("bicycle", "cycle_state.without_rider"),
    ("bicycle", "motorcycle_state.without_rider"),
    ("motorcycle", "vehicle_state.parked"),
    ("motorcycle", "cycle_state.without_rider"),
    ("motorcycle", "motorcycle_state.without_rider"),
]
