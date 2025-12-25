"""
Abstract base class for data loading in deployment.

Each task (classification, detection, segmentation, etc.) must implement
a concrete DataLoader that extends this base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, TypedDict

import torch


class SampleData(TypedDict, total=False):
    """
    Typed representation of a data sample handled by data loaders.

    Attributes:
        input: Raw input data such as images or point clouds.
        ground_truth: Labels or annotations if available.
        metadata: Additional information required for evaluation.
    """

    input: Any
    ground_truth: Any
    metadata: Dict[str, Any]


class BaseDataLoader(ABC):
    """
    Abstract base class for task-specific data loaders.

    This class defines the interface that all task-specific data loaders
    must implement. It handles loading raw data from disk and preprocessing
    it into a format suitable for model inference.
    """

    def __init__(self, config: Mapping[str, Any]):
        """
        Initialize data loader.

        Args:
            config: Configuration dictionary containing task-specific settings
        """
        self.config = config

    @abstractmethod
    def load_sample(self, index: int) -> SampleData:
        """
        Load a single sample from the dataset.

        Args:
            index: Sample index to load

        Returns:
            Dictionary containing raw sample data. Structure is task-specific,
            but should typically include:
            - Raw input data (image, point cloud, etc.)
            - Ground truth labels/annotations (if available)
            - Any metadata needed for evaluation

        Raises:
            IndexError: If index is out of range
            FileNotFoundError: If sample data files don't exist
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, sample: SampleData) -> torch.Tensor:
        """
        Preprocess raw sample data into model input format.

        Args:
            sample: Raw sample data returned by load_sample()

        Returns:
            Preprocessed tensor ready for model inference.
            Shape and format depend on the specific task.

        Raises:
            ValueError: If sample format is invalid
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_samples(self) -> int:
        """
        Get total number of samples in the dataset.

        Returns:
            Total number of samples available
        """
        raise NotImplementedError

    @abstractmethod
    def get_ground_truth(self, index: int) -> Mapping[str, Any]:
        """
        Get ground truth annotations for a specific sample.

        Args:
            index: Sample index whose annotations should be returned

        Returns:
            Dictionary containing task-specific ground truth data.
            Implementations should raise IndexError if the index is invalid.
        """
        raise NotImplementedError

    def get_shape_sample(self, index: int = 0) -> Any:
        """
        Return a representative sample used for export shape configuration.

        This method provides a consistent interface for exporters to obtain
        shape information without needing to know the internal structure of
        preprocessed inputs (e.g., whether it's a single tensor, tuple, or list).

        The default implementation:
        1. Loads a sample using load_sample()
        2. Preprocesses it using preprocess()
        3. If the result is a list/tuple, returns the first element
        4. Otherwise returns the preprocessed result as-is

        Subclasses can override this method to provide custom shape sample logic
        if the default behavior is insufficient.

        Args:
            index: Sample index to use (default: 0)

        Returns:
            A representative sample for shape configuration. Typically a torch.Tensor,
            but the exact type depends on the task-specific implementation.
        """
        sample = self.load_sample(index)
        preprocessed = self.preprocess(sample)

        # Handle nested structures: if it's a list/tuple, use first element for shape
        if isinstance(preprocessed, (list, tuple)):
            return preprocessed[0] if len(preprocessed) > 0 else preprocessed

        return preprocessed
