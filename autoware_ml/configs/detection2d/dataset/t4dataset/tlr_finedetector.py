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
    "BACKGROUND",
    "traffic_light",
    "pedestrian_traffic_light",
)

class_mappings = {
    "green": "traffic_light",
    "left-red": "traffic_light",
    "left-red-straight": "traffic_light",
    "red": "traffic_light",
    "red-right": "traffic_light",
    "red-straight": "traffic_light",
    "yellow": "traffic_light",
    "red-rightdiagonal": "traffic_light",
    "right-yellow": "traffic_light",
    "red-right-straight": "traffic_light",
    "leftdiagonal-red": "traffic_light",
    "unknown": "traffic_light",
    "red_right": "traffic_light",
    "red_left": "traffic_light",
    "red_straight_left": "traffic_light",
    "red_straight": "traffic_light",
    "crosswalk_red": "pedestrian_traffic_light",
    "crosswalk_green": "pedestrian_traffic_light",
    "crosswalk_unknown": "pedestrian_traffic_light",
    "green_straight": "traffic_light",
    "green_left": "traffic_light",
    "green_right": "traffic_light",
    "yellow_straight": "traffic_light",
    "yellow_left": "traffic_light",
    "yellow_right": "traffic_light",
    "yellow_straight_left": "traffic_light",
    "yellow_straight_right": "traffic_light",
    "yellow_straight_left_right": "traffic_light",
    "red_straight_right": "traffic_light",
    "red_straight_left_right": "traffic_light",
    "red_leftdiagonal": "traffic_light",
}
