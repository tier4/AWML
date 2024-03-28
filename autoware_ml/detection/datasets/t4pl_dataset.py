from autoware_ml.registry import DATASETS

from .t4dataset import T4Dataset


@DATASETS.register_module()
class T4PLDataset(T4Dataset):
    """T4 Pseudo Label Dataset."""

    # ref: https://github.com/tier4/tier4_autoware_msgs/blob/48ab51b908/tier4_perception_msgs/msg/object_recognition/Semantic.msg
    NameMapping = {
        "unknown": "unknown",
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "bicycle": "bicycle",
        "motorbike": "bicycle",
        "motorcycle": "bicycle",
        "pedestrian": "pedestrian",
        "animal": "animal",
    }
