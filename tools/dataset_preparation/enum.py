from enum import Enum


class DatasetInfoSplitKey(Enum):
    """Supported split names in data preparation."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TRAIN_VAL = "trainval"
    ALL = "all"

    def __str__(self):
        """String representation."""
        return self.value


class DatasetTask(Enum):
    """Supported dataset tasks in data preparation."""

    T4DETECTION3D = "t4_detection3d"
    T4DETECTION2D = "t4_detection2d"
    T4CLASSIFICATION2D = "t4_classification2d"

    def __str__(self):
        """String representation."""
        return self.value
