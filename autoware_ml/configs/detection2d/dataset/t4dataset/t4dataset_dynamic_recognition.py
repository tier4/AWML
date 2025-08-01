dataset_version_config_root = "autoware_ml/configs/detection2d/dataset/t4dataset/"
dataset_version_list = [
    "db_gsm8_v2",
    "db_jpntaxi_v1",
]

classes = (
    "unknown",
    "car",
    "truck",
    "bus",
    "trailer",
    "motorcycle",
    "pedestrian",
    "bicycle",
)

class_mappings = {
    # 'movable_object.barrier': 'barrier',
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.bus": "bus",
    # 'vehicle.construction': 'construction_vehicle',
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    # 'movable_object.trafficcone': 'traffic_cone',
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
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
    # "static_object.bicycle rack": "bicycle rack",
    # "static_object.bollard": "bollard",
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
}
