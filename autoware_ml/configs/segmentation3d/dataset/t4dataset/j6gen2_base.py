# dataset type setting
dataset_type = "T4Dataset"
info_train_file_name = "t4dataset_j6gen2_base_infos_train.pkl"
info_val_file_name = "t4dataset_j6gen2_base_infos_val.pkl"
info_test_file_name = "t4dataset_j6gen2_base_infos_test.pkl"

# dataset scene setting
dataset_version_config_root = "autoware_ml/configs/t4dataset/"
dataset_version_list = [
    "db_j6gen2_semaseg_v1",
]

camera_types = {}

# segmentation only uses lidar
input_modality = dict(use_lidar=True, use_camera=False)

# class setting
class_names = [
    "unpainted",
    "driveable_surface",
    "other_flat_surface",
    "sidewalk",
    "manmade",
    "vegetation",
    "car",
    "bus",
    "emergency_vehicle",
    "train",
    "truck",
    "tractor_unit",
    "semi_trailer",
    "construction_vehicle",
    "forklift",
    "kart",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "personal_mobility",
    "animal",
    "pushable_pullable",
    "traffic_cone",
    "stroller",
    "debris",
    "other_stuff",
    "noise",
    "ghost_point",
    "out_of_sync",
]

metainfo = dict(
    classes=class_names,
)

merge_objects = None
merge_type = None

# visualization
class_colors = {
    0: (0, 0, 0),  # Black
    1: (0, 255, 255),  # Cyan / Aqua
    2: (233, 233, 229),  # Light Beige/Grey
    3: (110, 110, 110),  # Dark Grey
    4: (0, 175, 0),  # Green
    5: (232, 35, 244),  # Magenta
    6: (255, 158, 0),  # Orange
    7: (0, 0, 230),  # Blue
    8: (255, 0, 0),  # Bright Red
    9: (255, 127, 80),  # Coral
    10: (160, 60, 60),  # Brownish Red
    11: (255, 140, 0),  # Dark Orange
    12: (255, 215, 0),  # Gold
    13: (220, 20, 60),  # Crimson
    14: (255, 61, 99),  # Reddish Pink
    15: (230, 230, 0),  # Yellow
    16: (128, 128, 128),  # Medium Grey
    17: (255, 0, 255),  # Bright Fuchsia
    18: (47, 79, 79),  # Dark Slate Grey
    19: (139, 69, 19),  # Saddle Brown
    20: (218, 165, 32),  # Goldenrod
    21: (128, 0, 128),  # Purple
    22: (135, 206, 235),  # Sky Blue
    23: (0, 0, 128),  # Navy Blue
    24: (170, 170, 170),  # Light Grey
    25: (205, 133, 63),  # Peru
    26: (255, 99, 71),  # Tomato Red
    27: (240, 240, 240),  # Ghost White
    28: (50, 50, 50),  # Very Dark Grey
}

filter_attributes = None
