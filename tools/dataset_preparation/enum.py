from enum import Enum 


class DatasetInfoSplitKey(Enum):
    """ Supported split names in data preparation. """

    TRAIN = "train"
    VAL = "val"
    TEST = "test" 
    TRAIN_VAL = "trainval"
    ALL = "all"

    def __str__(self):
        """ String representation. """
        return self.value


class Task(Enum):
    """ Supported types in data preparation. """
    
    DETECTION3D = "detection3d"
    DETECTION2D = "detection2d"
    CLASSIFICATION2D = "classification2d"
    
    def __str__(self):
        """ String representation. """
        return self.value

