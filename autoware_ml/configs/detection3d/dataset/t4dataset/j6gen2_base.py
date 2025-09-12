custom_imports = dict(
    imports=[
        "autoware_ml.detection3d.datasets.t4dataset",
        "autoware_ml.detection3d.evaluation.t4metric.t4metric",
        "autoware_ml.detection3d.evaluation.t4metric.t4metric_v2",
    ]
)

# dataset type setting
dataset_type = "T4Dataset"
info_train_file_name = "t4dataset_j6gen2_base_infos_train.pkl"
info_val_file_name = "t4dataset_j6gen2_base_infos_val.pkl"
info_test_file_name = "t4dataset_j6gen2_base_infos_test.pkl"

# dataset scene setting
dataset_version_config_root = "autoware_ml/configs/t4dataset/"
dataset_version_list = [
    "db_j6gen2_v1",
    "db_j6gen2_v2",
    "db_j6gen2_v3",
    "db_j6gen2_v4",
    "db_j6gen2_v5",
    "db_largebus_v1",
    "db_largebus_v2",
]

dataset_test_groups = {
    "db_j6gen2": "t4dataset_j6gen2_infos_test.pkl",
    "db_largebus": "t4dataset_largebus_infos_test.pkl",
    "db_j6_gen2_base": "t4dataset_j6gen2_base_infos_test.pkl",
}

# dataset format setting
data_prefix = dict(
    pts="",
    CAM_FRONT="",
    CAM_FRONT_LEFT="",
    CAM_FRONT_RIGHT="",
    CAM_BACK="",
    CAM_BACK_RIGHT="",
    CAM_BACK_LEFT="",
    sweeps="",
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
    # DBv2.0 and DBv3.0
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
    "vehicle.ambulance": "car",  # Define vehicle.ambulance as car since vehicle.emergency (ambulance & police) is defined as car
    "vehicle.bicycle": "bicycle",
    "vehicle.bus": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "truck",
    "vehicle.fire": "truck",
    "vehicle.motorcycle": "bicycle",
    "vehicle.police": "car",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
    # DBv1.3
    "ambulance": "car",
    "kart": "car",
    "wheelchair": "pedestrian",
    "personal_mobility": "pedestrian",
    "fire_truck": "truck",
    "semi_trailer": "trailer",
    "tractor_unit": "truck",
    "construction_vehicle": "truck",
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

evaluator_metric_configs = dict(
    evaluation_task="detection",
    target_labels=class_names,
    center_distance_bev_thresholds=[0.5, 1.0, 2.0, 4.0],
    # plane_distance_thresholds is required for the pass fail evaluation
    plane_distance_thresholds=[2.0, 4.0],
    iou_2d_thresholds=None,
    iou_3d_thresholds=None,
    label_prefix="autoware",
    max_distance=121.0,
    min_distance=-121.0,
    min_point_numbers=0,
)
