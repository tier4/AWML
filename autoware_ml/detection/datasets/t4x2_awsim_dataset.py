from autoware_ml.registry import DATASETS

from .t4dataset import T4Dataset


@DATASETS.register_module()
class T4X2AwsimDataset(T4Dataset):
    """T4X2AwsimDataset Synthetic Dataset."""

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
        "street_asset": "street_asset",
        "bicycle_without_rider": "bicycle_without_rider",
        "motorbike_without_rider": "motorbike_without_rider",
    }
