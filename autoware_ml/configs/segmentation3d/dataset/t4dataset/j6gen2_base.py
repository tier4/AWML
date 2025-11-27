custom_imports = dict(
    imports=[
        "autoware_ml.detection3d.datasets.t4dataset",
    ],
    allow_failed_imports=False,
)

# dataset type setting
dataset_type = "T4Dataset"
info_train_file_name = "t4dataset_j6gen2_base_infos_train.pkl"
info_val_file_name = "t4dataset_j6gen2_base_infos_val.pkl"
info_test_file_name = "t4dataset_j6gen2_base_infos_test.pkl"

# dataset scene setting
dataset_version_config_root = "autoware_ml/configs/t4dataset/"
dataset_version_list = [
    "db_j6gen2_lidarseg_v1",
]

dataset_test_groups = {
    "db_j6gen2": "t4dataset_j6gen2_base_infos_test.pkl",
}

# dataset format setting
data_prefix = dict(
    pts="",
    pts_semantic_mask="",
)
camera_types = {}

# segmentation only uses lidar
input_modality = dict(use_lidar=True, use_camera=False)

# class setting
class_names = [
    "car",
    "emergency_vehicle",
    "motorcycle",
    "tractor_unit",
    "semi_trailer",
    "truck",
    "bicycle",
    "bus",
    "forklift",
    "kart",
    "construction_vehicle",
    "train",
    "pedestrian",
    "personal_mobility",
    "stroller",
    "animal",
    "traffic_cone",
    "debris",
    "pushable_pullable",
    "vegetation",
    "manmade",
    "other_stuff",
    "driveable_surface",
    "sidewalk",
    "other_flat_surface",
    "noise",
    "ghost_point",
    "out_of_sync",
]

num_class = len(class_names)

name_mapping = {
    "car": "car",
    "emergency_vehicle": "emergency_vehicle",
    "motorcycle": "motorcycle",
    "tractor_unit": "tractor_unit",
    "semi_trailer": "semi_trailer",
    "truck": "truck",
    "bicycle": "bicycle",
    "bus": "bus",
    "forklift": "forklift",
    "kart": "kart",
    "construction_vehicle": "construction_vehicle",
    "train": "train",
    "pedestrian": "pedestrian",
    "personal_mobility": "personal_mobility",
    "stroller": "stroller",
    "animal": "animal",
    "traffic_cone": "traffic_cone",
    "debris": "debris",
    "pushable_pullable": "pushable_pullable",
    "vegetation": "vegetation",
    "manmade": "manmade",
    "other_stuff": "other_stuff",
    "driveable_surface": "driveable_surface",
    "sidewalk": "sidewalk",
    "other_flat_surface": "other_flat_surface",
    "noise": "noise",
    "ghost_point": "ghost_point",
    "out_of_sync": "out_of_sync",
}

metainfo = dict(
    classes=class_names,
)

merge_objects = None
merge_type = None

# visualization
class_colors = {
    "car": (30, 144, 255),  # Dodger Blue
    "emergency_vehicle": (255, 0, 0),  # Red
    "motorcycle": (100, 0, 30),  # Dark Red
    "tractor_unit": (180, 0, 255),  # Purple
    "semi_trailer": (0, 255, 255),  # Cyan
    "truck": (140, 0, 255),  # Purple
    "bicycle": (255, 0, 30),  # Red
    "bus": (111, 255, 111),  # Light Green
    "forklift": (255, 165, 0),  # Orange
    "kart": (255, 192, 203),  # Pink
    "construction_vehicle": (255, 255, 0),  # Yellow
    "train": (70, 130, 180),  # Steel Blue
    "pedestrian": (255, 200, 200),  # Light Pink
    "personal_mobility": (255, 150, 150),  # Pink
    "stroller": (200, 100, 100),  # Dark Pink
    "animal": (150, 75, 0),  # Brown
    "traffic_cone": (120, 120, 120),  # Gray
    "debris": (80, 80, 80),  # Dark Gray
    "pushable_pullable": (100, 100, 100),  # Gray
    "vegetation": (0, 255, 0),  # Green
    "manmade": (128, 128, 128),  # Gray
    "other_stuff": (60, 60, 60),  # Dark Gray
    "driveable_surface": (50, 50, 50),  # Very Dark Gray
    "sidewalk": (150, 150, 150),  # Light Gray
    "other_flat_surface": (100, 100, 100),  # Gray
    "noise": (0, 0, 0),  # Black
    "ghost_point": (128, 0, 128),  # Purple
    "out_of_sync": (255, 0, 255),  # Magenta
}

filter_attributes = None
