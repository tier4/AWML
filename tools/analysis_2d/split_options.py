from enum import Enum


class SplitOptions(str, Enum):
    """Available split options."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __str__(self) -> str:
        """String representation."""
        return str.__str__(self)
