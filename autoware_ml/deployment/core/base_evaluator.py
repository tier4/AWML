"""
Abstract base class for model evaluation in deployment.
Each task (classification, detection, segmentation, etc.) must implement
a concrete Evaluator that extends this base class to compute task-specific metrics.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from .base_data_loader import BaseDataLoader


class BaseEvaluator(ABC):
    """
    Abstract base class for task-specific evaluators.
    This class defines the interface that all task-specific evaluators
    must implement. It handles running inference on a dataset and computing
    evaluation metrics appropriate for the task.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator.
        Args:
            config: Configuration dictionary containing evaluation settings
        """
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        model_path: str,
        data_loader: BaseDataLoader,
        num_samples: int,
        backend: str = "pytorch",
        device: str = "cpu",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full evaluation on a model.
        Args:
            model_path: Path to model checkpoint/weights
            data_loader: DataLoader for loading samples
            num_samples: Number of samples to evaluate
            backend: Backend to use ('pytorch', 'onnx', 'tensorrt')
            device: Device to run inference on
            verbose: Whether to print detailed progress
        Returns:
            Dictionary containing evaluation metrics. The exact metrics
            depend on the task, but should include:
            - Primary metric(s) for the task
            - Per-class metrics (if applicable)
            - Inference latency statistics
            - Any other relevant metrics
        Example:
            For classification:
            {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "per_class_accuracy": {...},
                "confusion_matrix": [...],
                "avg_latency_ms": 5.2,
            }
            For detection:
            {
                "mAP": 0.72,
                "mAP_50": 0.85,
                "mAP_75": 0.68,
                "per_class_ap": {...},
                "avg_latency_ms": 15.3,
            }
        """
        pass

    @abstractmethod
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Pretty print evaluation results.
        Args:
            results: Results dictionary returned by evaluate()
        """
        pass

    @abstractmethod
    def verify(
        self,
        ref_backend: str,
        ref_device: str,
        ref_path: str,
        test_backend: str,
        test_device: str,
        test_path: str,
        data_loader: BaseDataLoader,
        num_samples: int = 1,
        tolerance: float = 0.1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Verify exported models using scenario-based verification.

        This method compares outputs from a reference backend against a test backend
        as specified by the verification scenarios. This is a more flexible approach
        than the legacy verify() method which compares all available backends.

        Args:
            ref_backend: Reference backend name ('pytorch' or 'onnx')
            ref_device: Device for reference backend (e.g., 'cpu', 'cuda:0')
            ref_path: Path to reference model (checkpoint for pytorch, model path for onnx)
            test_backend: Test backend name ('onnx' or 'tensorrt')
            test_device: Device for test backend (e.g., 'cpu', 'cuda:0')
            test_path: Path to test model (model path for onnx, engine path for tensorrt)
            data_loader: Data loader for test samples
            num_samples: Number of samples to verify
            tolerance: Maximum allowed difference for verification to pass
            verbose: Whether to print detailed output

        Returns:
            Dictionary containing verification results:
            {
                'sample_0': bool (passed/failed),
                'sample_1': bool (passed/failed),
                ...
                'summary': {'passed': int, 'failed': int, 'total': int}
            }
        """
        pass

    def compute_latency_stats(self, latencies: list) -> Dict[str, float]:
        """
        Compute latency statistics from a list of latency measurements.
        Args:
            latencies: List of latency values in milliseconds
        Returns:
            Dictionary with latency statistics
        """
        if not latencies:
            return {
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "median_ms": 0.0,
            }

        latencies_array = np.array(latencies)

        return {
            "mean_ms": float(np.mean(latencies_array)),
            "std_ms": float(np.std(latencies_array)),
            "min_ms": float(np.min(latencies_array)),
            "max_ms": float(np.max(latencies_array)),
            "median_ms": float(np.median(latencies_array)),
        }

    def format_latency_stats(self, stats: Dict[str, float]) -> str:
        """
        Format latency statistics as a readable string.
        Args:
            stats: Latency statistics dictionary
        Returns:
            Formatted string
        """
        return (
            f"Latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms "
            f"(min: {stats['min_ms']:.2f}, max: {stats['max_ms']:.2f}, "
            f"median: {stats['median_ms']:.2f})"
        )
