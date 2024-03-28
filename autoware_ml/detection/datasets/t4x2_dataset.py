from autoware_ml.registry import DATASETS

from .t4dataset import T4Dataset


@DATASETS.register_module()
class T4X2Dataset(T4Dataset):
    """T4X2Dataset Dataset."""

    NameMapping = {
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
