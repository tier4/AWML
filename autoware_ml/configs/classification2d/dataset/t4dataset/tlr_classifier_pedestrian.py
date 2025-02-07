dataset_version_config_root = "autoware_ml/configs/t4dataset/"
dataset_version_list = [
    "db_tlr_v1",
    "db_tlr_v2",
    "db_tlr_v3",
    "db_tlr_v4",
    "db_tlr_v5",
    "db_tlr_v6",
]

classes = (
    "crosswalk_red",
    "crosswalk_green",
    "crosswalk_unknown",
)
class_mappings = {
    "crosswalk_red": "crosswalk_red",
    "crosswalk_green": "crosswalk_green",
    "crosswalk_unknown": "crosswalk_unknown",
}
class_mappings = {
    "crosswalk_red": "crosswalk_red",
    "crosswalk_green": "crosswalk_green",
    "crosswalk_unknown": "crosswalk_unknown",
    # skip the following classes if present like in AWMLDetection2d
    "green": "SKIP_CLASS",
    "left-red": "SKIP_CLASS",
    "left-red-straight": "SKIP_CLASS",
    "red": "SKIP_CLASS",
    "red-right": "SKIP_CLASS",
    "red-straight": "SKIP_CLASS",
    "yellow": "SKIP_CLASS",
    "red-rightdiagonal": "SKIP_CLASS",
    "right-yellow": "SKIP_CLASS",
    "red-right-straight": "SKIP_CLASS",
    "leftdiagonal-red": "SKIP_CLASS",
    "unknown": "SKIP_CLASS",
    "red_right": "SKIP_CLASS",
    "red_left": "SKIP_CLASS",
    "red_straight_left": "SKIP_CLASS",
    "red_straight": "SKIP_CLASS",
    "green_straight": "SKIP_CLASS",
    "green_left": "SKIP_CLASS",
    "green_right": "SKIP_CLASS",
    "yellow_straight": "SKIP_CLASS",
    "yellow_left": "SKIP_CLASS",
    "yellow_right": "SKIP_CLASS",
    "yellow_straight_left": "SKIP_CLASS",
    "yellow_straight_right": "SKIP_CLASS",
    "yellow_straight_left_right": "SKIP_CLASS",
    "red_straight_right": "SKIP_CLASS",
    "red_straight_left_right": "SKIP_CLASS",
    "red_leftdiagonal": "SKIP_CLASS",
}
