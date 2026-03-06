dataset_version_config_root = "autoware_ml/configs/t4dataset/"
dataset_version_list = [
    "db_tlr_v1",
    "db_tlr_v2",
    "db_tlr_v3",
    "db_tlr_v4",
    "db_tlr_v5",
    "db_tlr_v6",  # does not have any pedestrian tlr data
    "db_tlr_v7",  # does not have any pedestrian tlr data
    "db_tlr_semseg_v1",
]

classes = (
    "crosswalk_red",
    "crosswalk_green",
    "crosswalk_unknown",
)

class_mappings = {
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
    # Skip the following semantic TLR if present
    "traffic_light_back": "SKIP_CLASS",
    "crosswalk_light_back": "SKIP_CLASS",
    # Bulb boxes
    "red_bulb": "SKIP_CLASS",
    "green_bulb": "SKIP_CLASS",
    "yellow_bulb": "SKIP_CLASS",
    "red_left_arrow_bulb": "SKIP_CLASS",
    "red_right_arrow_bulb": "SKIP_CLASS",
    "red_straight_arrow_bulb": "SKIP_CLASS",
    "red_up_left_arrow_bulb": "SKIP_CLASS",
    "red_up_right_arrow_bulb": "SKIP_CLASS",
    "red_arrow_unknown_bulb": "SKIP_CLASS",
    "green_left_arrow_bulb": "SKIP_CLASS",
    "green_right_arrow_bulb": "SKIP_CLASS",
    "green_straight_arrow_bulb": "SKIP_CLASS",
    "green_up_left_arrow_bulb": "SKIP_CLASS",
    "green_up_right_arrow_bulb": "SKIP_CLASS",
    "green_arrow_unknown_bulb": "SKIP_CLASS",
    "yellow_left_arrow_bulb": "SKIP_CLASS",
    "yellow_right_arrow_bulb": "SKIP_CLASS",
    "yellow_straight_arrow_bulb": "SKIP_CLASS",
    "yellow_up_left_arrow_bulb": "SKIP_CLASS",
    "yellow_up_right_arrow_bulb": "SKIP_CLASS",
    "yellow_arrow_unknown_bulb": "SKIP_CLASS",
}
