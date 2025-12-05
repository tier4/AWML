# dataset type setting
dataset_type = "T4Dataset"
info_train_file_name = "t4dataset_j6gen2_base_infos_train.pkl"
info_val_file_name = "t4dataset_j6gen2_base_infos_val.pkl"
info_test_file_name = "t4dataset_j6gen2_base_infos_test.pkl"

# dataset scene setting
dataset_version_config_root = "autoware_ml/configs/t4dataset/"
dataset_version_list = [
    "db_semseg_demo",
]

camera_types = {}

# segmentation only uses lidar
input_modality = dict(use_lidar=True, use_camera=False)

# class setting
class_names = [
    "unknown",
    "driveable_surface",
    "manmade",
    "other_flat_surface",
    "vegetation",
    "sidewalk",
    "car",
    "pedestrian",
    "emergency_vehicle",
    "truck",
    "tractor_unit",
    "semi_trailer",
    "bus",
    "bicycle",
    "motorcycle",
    "construction_vehicle",
    "other_stuff",
    "noise",
    "traffic_cone",
    "debris",
    "forklift",
    "kart",
    "stroller",
    "personal_mobility",
    "pushable_pullable",
    "animal",
    "train",
    "ghost_point",
    "out_of_sync",
]

num_class = len(class_names)

metainfo = dict(
    classes=class_names,
)

merge_objects = None
merge_type = None

# visualization
class_colors = {
    "unknown": (0, 0, 0),  # Black
    "driveable_surface": (0, 255, 255),  # Cyan / Aqua
    "manmade": (233, 233, 229),  # Light Beige/Grey
    "other_flat_surface": (110, 110, 110),  # Dark Grey
    "vegetation": (0, 175, 0),  # Green
    "sidewalk": (232, 35, 244),  # Magenta
    "car": (255, 158, 0),  # Orange
    "pedestrian": (0, 0, 230),  # Blue
    "emergency_vehicle": (255, 0, 0),  # Bright Red
    "truck": (255, 127, 80),  # Coral
    "tractor_unit": (160, 60, 60),  # Brownish Red
    "semi_trailer": (255, 140, 0),  # Dark Orange
    "bus": (255, 215, 0),  # Gold
    "bicycle": (220, 20, 60),  # Crimson
    "motorcycle": (255, 61, 99),  # Reddish Pink
    "construction_vehicle": (230, 230, 0),  # Yellow
    "other_stuff": (128, 128, 128),  # Medium Grey
    "noise": (255, 0, 255),  # Bright Fuchsia
    "traffic_cone": (47, 79, 79),  # Dark Slate Grey
    "debris": (139, 69, 19),  # Saddle Brown
    "forklift": (218, 165, 32),  # Goldenrod
    "kart": (128, 0, 128),  # Purple
    "stroller": (135, 206, 235),  # Sky Blue
    "personal_mobility": (0, 0, 128),  # Navy Blue
    "pushable_pullable": (170, 170, 170),  # Light Grey
    "animal": (205, 133, 63),  # Peru
    "train": (255, 99, 71),  # Tomato Red
    "ghost_point": (50, 50, 50),  # Very Dark Grey
    "out_of_sync": (255, 255, 255),  # White
}

filter_attributes = None
