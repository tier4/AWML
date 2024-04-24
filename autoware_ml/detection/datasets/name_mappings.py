from typing import Dict

T4X2 = {
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
    "static_object.bicycle rack": "bicycle rack",
    "static_object.bollard": "bollard",
    "vehicle.ambulance": "truck",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "truck",
    "vehicle.fire": "truck",
    "vehicle.motorcycle": "bicycle",
    "vehicle.police": "car",
    "vehicle.trailer": "truck",
    "vehicle.truck": "truck",
}


T4XX1 = {
    # DBv1.0
    "vehicle.car": "car",
    "vehicle.construction": "truck",
    "vehicle.emergency (ambulance & police)": "car",
    "vehicle.motorcycle": "bicycle",
    "vehicle.trailer": "truck",
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
    "trailer": "truck",
    "motorcycle": "bicycle",
    "bicycle": "bicycle",
    "police_car": "car",
    "pedestrian": "pedestrian",
    "police_officer": "pedestrian",
    "forklift": "car",
    "construction_worker": "pedestrian",
    "stroller": "pedestrian",
}

def get_mapping(dataset_version: str) -> Dict[str, str]:
    if dataset_version in ["t4xx1", "t4xx1_uc2"]:
        return T4XX1
    elif dataset_version == "t4x2":
        return T4X2
    else:
        raise ValueError(f"not supported dataset_version: {dataset_version}")
