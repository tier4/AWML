custom_imports = dict(
    imports=[
        "autoware_ml.detection3d.datasets.t4dataset_full_categories",
        "autoware_ml.detection3d.evaluation.t4metric.t4metric",
        "autoware_ml.detection3d.evaluation.t4metric.t4metric_v2",
    ]
)

# dataset type — uses label-remapping subclass so pkl can stay at 5-class format
dataset_type = "T4DatasetFullCategories"

# Pickle files at /mnt/qnapdata/internal/t4datasets/info/kang/lidardet/
# These are the j6gen2 infos (NOT j6gen2_base_infos).
# They store gt_nusc_name for every instance, which this dataset class uses to
# re-assign labels according to the broader class set below.
info_train_file_name = "t4dataset_j6gen2_infos_train.pkl"
info_val_file_name = "t4dataset_j6gen2_infos_val.pkl"
info_test_file_name = "t4dataset_j6gen2_infos_test.pkl"

# dataset scene setting
dataset_version_list = [
    "db_j6gen2_v1",
    "db_j6gen2_v2",
    "db_j6gen2_v3",
    "db_j6gen2_v4",
    "db_j6gen2_v5",
    "db_j6gen2_v6",
    "db_j6gen2_v7",
    "db_j6gen2_v8",
    "db_j6gen2_v9",
    "db_largebus_v1",
    "db_largebus_v2",
    "db_largebus_v3",
]

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

# ---------------------------------------------------------------------------
# 19-class set — aligned to PTv3 semseg taxonomy where possible.
# PTv3 semseg class index shown in parentheses.
#
# Notable decisions vs PTv3:
#   construction_worker → pedestrian(17)  [no separate PTv3 class]
#   tractor             → construction_vehicle(12)
#   wheelchair          → personal_mobility(18)
#   barrier             → kept as "barrier" bbox class; in CURRENT PTv3 semseg
#                         annotations barrier points are labeled "manmade"(3),
#                         NOT pushable_pullable(20). May change in future PTv3 version.
# ---------------------------------------------------------------------------
class_names = [
    "car",                  # 0  PTv3: 5
    "truck",                # 1  PTv3: 9   (truck, fire_truck)
    "bus",                  # 2  PTv3: 6
    "construction_vehicle", # 3  PTv3: 12  (construction_vehicle, tractor)
    "tractor_unit",         # 4  PTv3: 10
    "semi_trailer",         # 5  PTv3: 11  (semi_trailer, trailer)
    "train",                # 6  PTv3: 8
    "emergency_vehicle",    # 7  PTv3: 7   (ambulance, police_car)
    "forklift",             # 8  PTv3: 13
    "kart",                 # 9  PTv3: 14
    "other_vehicle",        # 10 no PTv3 match
    "bicycle",              # 11 PTv3: 16  (all — parked filter removed)
    "motorcycle",           # 12 PTv3: 15  (all — parked filter removed)
    "pedestrian",           # 13 PTv3: 17  (pedestrian, construction_worker, police_officer, other_pedestrian)
    "personal_mobility",    # 14 PTv3: 18  (personal_mobility, wheelchair)
    "stroller",             # 15 PTv3: 22
    "animal",               # 16 PTv3: 19
    "traffic_cone",         # 17 PTv3: 21
    "barrier",              # 18 → PTv3 "manmade"(3) in current annotation version
    "pushable_pullable",    # 19 PTv3: 20
]
num_class = len(class_names)
metainfo = dict(classes=class_names)

# ---------------------------------------------------------------------------
# extended_name_mapping: maps gt_nusc_name (raw annotation name as stored in
# the pkl, e.g. "semi_trailer", "ambulance") to one of the class_names above.
# Names absent here or mapped to None receive label -1 and are filtered out.
# ---------------------------------------------------------------------------
# Maps every gt_nusc_name string actually present in the pkl (verified from data)
# to one of the class_names above, or None to ignore.
extended_name_mapping = {
    # vehicles — large
    "car":                  "car",
    "truck":                "truck",
    "fire_truck":           "truck",
    "bus":                  "bus",
    "construction_vehicle": "construction_vehicle",
    "tractor":              "construction_vehicle",
    "tractor_unit":         "tractor_unit",
    "semi_trailer":         "semi_trailer",
    "trailer":              "semi_trailer",
    "train":                "train",
    "ambulance":            "emergency_vehicle",
    "police_car":           "emergency_vehicle",
    "forklift":             "forklift",
    "kart":                 "kart",
    "other_vehicle":        "other_vehicle",
    # two-wheelers (parked/without-rider filter removed)
    "bicycle":              "bicycle",
    "motorcycle":           "motorcycle",
    # pedestrians (construction_worker has no separate PTv3 class)
    "pedestrian":           "pedestrian",
    "other_pedestrian":     "pedestrian",
    "police_officer":       "pedestrian",
    "construction_worker":  "pedestrian",
    # mobility aids / small ride-ons
    "stroller":             "stroller",
    "wheelchair":           "personal_mobility",
    "personal_mobility":    "personal_mobility",
    # fauna
    "animal":               "animal",
    # movable objects
    "traffic_cone":         "traffic_cone",
    "barrier":              "barrier",       # → PTv3 "manmade"(3) in current annotations
    "pushable_pullable":    "pushable_pullable",  # → PTv3 "pushable_pullable"(20)
    # ignored — no PTv3 counterpart and too few annotations
    "construction_sign":    None,
    "construction sign":    None,
}

# ---------------------------------------------------------------------------
# name_mapping for T4Metric evaluator.
# Maps ALL raw DB annotation names (across every DB version) to the new class names.
# This is used at evaluation time when reading ground-truth from the annotation files.
# ---------------------------------------------------------------------------
name_mapping = {
    # DBv1.0 / vehicle.* namespace
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.ambulance": "emergency_vehicle",
    "vehicle.police": "emergency_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "semi_trailer",
    "vehicle.truck": "truck",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus": "bus",
    "vehicle.fire": "truck",
    # DBv1.1 / UCv2.0 flat names
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "trailer": "semi_trailer",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "police_car": "emergency_vehicle",
    "ambulance": "emergency_vehicle",
    "fire_truck": "truck",
    "pedestrian": "pedestrian",
    "police_officer": "pedestrian",
    "forklift": "forklift",
    "construction_worker": "pedestrian",
    "stroller": "stroller",
    "wheelchair": "personal_mobility",
    "personal_mobility": "personal_mobility",
    # DBv1.3 additional names
    "kart": "kart",
    "semi_trailer": "semi_trailer",
    "tractor_unit": "tractor_unit",
    "tractor": "construction_vehicle",
    "construction_vehicle": "construction_vehicle",
    "train": "train",
    "other_vehicle": "other_vehicle",
    "other_pedestrian": "pedestrian",
    # movable objects
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.traffic_cone": "traffic_cone",
    "movable_object.pushable_pullable": "pushable_pullable",
    "movable_object.debris": None,
    # pedestrians (DBv1.0 namespace)
    "pedestrian.adult": "pedestrian",
    "pedestrian.child": "pedestrian",
    "pedestrian.construction_worker": "pedestrian",
    "pedestrian.personal_mobility": "personal_mobility",
    "pedestrian.police_officer": "pedestrian",
    "pedestrian.stroller": "stroller",
    "pedestrian.wheelchair": "personal_mobility",
    # misc
    "animal": "animal",
    "construction_sign": None,
    "construction sign": None,
    "static_object.bicycle_rack": None,
    "static_object.bicycle rack": None,
    "static_object.bollard": None,
}

# No class merging — classes are kept separate for downstream pseudo-label use.
merge_objects = []
merge_type = None

# No attribute-based filtering for bicycle/motorcycle:
# We want ALL instances (including parked/without-rider) since the goal is
# point-cloud labeling, not just active-object detection.
filter_attributes = []

evaluator_metric_configs = dict(
    evaluation_task="detection",
    target_labels=class_names,
    center_distance_bev_thresholds=[0.5, 1.0, 2.0, 4.0],
    plane_distance_thresholds=[2.0, 4.0],
    iou_2d_thresholds=None,
    iou_3d_thresholds=None,
    label_prefix="autoware",
    min_distance=[0.0, 50.0, 90.0, 0.0],
    max_distance=[50.0, 90.0, 121.0, 121.0],
    min_point_numbers=0,
    matching_class_agnostic_fps=False,
)
