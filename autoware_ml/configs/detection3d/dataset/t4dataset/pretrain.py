custom_imports = dict(
    imports=[
        "autoware_ml.detection3d.datasets.t4dataset",
        "autoware_ml.detection3d.evaluation.t4metric.t4metric",
        "autoware_ml.detection3d.evaluation.t4metric.t4metric_v2",
    ]
)

# dataset type setting
dataset_type = "T4Dataset"
info_train_file_name = "t4dataset_pretrain_infos_train.pkl"
info_val_file_name = "t4dataset_pretrain_infos_val.pkl"
info_test_file_name = "t4dataset_pretrain_infos_test.pkl"

# dataset scene setting
dataset_version_config_root = "autoware_ml/configs/t4dataset/"
dataset_version_list = [
    "pseudo_j6_v1",
    "pseudo_j6_v2",
]

# dataset format setting
data_prefix = dict(pts="", sweeps="")
camera_types = {
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
}

# class setting, for labels, they all already remapped when they're generated
name_mapping = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "bicycle": "bicycle",
    "pedestrian": "pedestrian",
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
