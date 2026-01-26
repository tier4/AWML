custom_imports = dict(
    imports=[
        "autoware_ml.segmentation3d.datasets.t4dataset",
        "autoware_ml.segmentation3d.datasets.transforms",
    ]
)

# dataset type setting
dataset_type = "T4SegDataset"
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
base_class_names = [
    "unpainted",
    "drivable_surface",
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
base_palette = [
    [0, 0, 0],  # Black - unpainted
    [255, 0, 255],  # Magenta - drivable_surface
    [139, 137, 137],  # Grey - other_flat_surface
    [75, 0, 75],  # Dark Purple - sidewalk
    [230, 230, 250],  # Lavender - manmade
    [0, 175, 0],  # Green - vegetation
    [0, 150, 245],  # Blue - car
    [255, 255, 0],  # Yellow - bus
    [199, 21, 133],  # Medium Violet Red - emergency_vehicle
    [0, 139, 139],  # Dark Cyan - train
    [160, 32, 240],  # Purple - truck
    [184, 134, 11],  # Dark Goldenrod - tractor_unit
    [135, 60, 0],  # Brown - semi_trailer
    [0, 255, 255],  # Cyan - construction_vehicle
    [128, 128, 0],  # Olive - forklift
    [124, 252, 0],  # Lawn Green - kart
    [255, 127, 0],  # Orange - motorcycle
    [255, 192, 203],  # Pink - bicycle
    [255, 0, 0],  # Red - pedestrian
    [210, 105, 30],  # Chocolate - personal_mobility
    [189, 183, 107],  # Dark Khaki - animal
    [153, 50, 204],  # Dark Orchid - pushable_pullable
    [255, 240, 150],  # Light Yellow - traffic_cone
    [70, 130, 180],  # Steel Blue - stroller
    [210, 180, 140],  # Tan - debris
    [188, 143, 143],  # Rosy Brown - other_stuff
    [64, 224, 208],  # Turquoise - noise
    [245, 245, 220],  # Beige - ghost_point
    [50, 50, 50],  # Very Dark Grey - out_of_sync
]

metainfo = dict(
    base_class_names=base_class_names,
    base_palette=base_palette,
)

merge_objects = None
merge_type = None

filter_attributes = None
