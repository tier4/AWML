#!/usr/bin/env python3

import argparse
import os
import pickle
import traceback
from typing import Any, Dict, List, Optional, Union

from mmengine.logging import MMLogger

from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)


class CalibrationVisualizer:
    """
    A comprehensive tool for visualizing LiDAR-camera calibration data.
    This class provides functionality to load calibration data from info.pkl files
    and generate visualizations using the CalibrationClassificationTransform.
    Attributes:
        transform: The calibration classification transform instance
        data_root: Root directory for data files
        output_dir: Directory for saving visualizations
        logger: MMLogger instance for logging
    """

    def __init__(self, data_root: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize the CalibrationVisualizer.
        Args:
            data_root: Root directory for data files. If None, absolute paths are used.
            output_dir: Directory for saving visualizations. If None, no visualizations are saved.
        """
        self.data_root = data_root
        self.output_dir = output_dir
        self.transform = None
        self.logger = MMLogger.get_instance(name="calibration_visualizer")
        self._initialize_transform()

    def _initialize_transform(self) -> None:
        """Initialize the CalibrationClassificationTransform with appropriate parameters."""
        self.transform = CalibrationClassificationTransform(
            mode="test",
            undistort=True,
            data_root=self.data_root,
            projection_vis_dir=self.output_dir,
            results_vis_dir=None,
            enable_augmentation=False,
        )

    def load_info_pkl(self, info_pkl_path: str) -> List[Dict[str, Any]]:
        """
        Load and parse info.pkl file.
        Args:
            info_pkl_path: Path to the info.pkl file.
        Returns:
            List of sample dictionaries.
        Raises:
            FileNotFoundError: If info.pkl file doesn't exist.
            ValueError: If data format is unexpected.
        """
        if not os.path.exists(info_pkl_path):
            raise FileNotFoundError(f"Info.pkl file not found: {info_pkl_path}")

        try:
            with open(info_pkl_path, "rb") as f:
                info_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load info.pkl file: {e}")

        return self._extract_samples_from_data(info_data)

    def _extract_samples_from_data(self, info_data: Union[Dict, List]) -> List[Dict[str, Any]]:
        """
        Extract sample list from info.pkl data format.
        Args:
            info_data: Raw data from info.pkl file.
        Returns:
            List of sample dictionaries.
        Raises:
            ValueError: If data format is not supported.
        """
        if isinstance(info_data, dict):
            if "data_list" in info_data:
                samples_list = info_data["data_list"]
                self.logger.info(f"Loaded {len(samples_list)} samples from info.pkl")
            else:
                raise ValueError(f"Expected 'data_list' key in info_data, found keys: {list(info_data.keys())}")
        else:
            raise ValueError(f"Expected dict format, got {type(info_data)}")

        return samples_list

    def _validate_sample_structure(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample has the required structure.
        Args:
            sample: Sample dictionary to validate.
        Returns:
            True if sample is valid, False otherwise.
        """
        required_keys = ["image", "lidar_points"]
        return all(key in sample for key in required_keys)

    def process_single_sample(self, sample: Dict[str, Any], sample_idx: int) -> Optional[Dict[str, Any]]:
        """
        Process a single sample using the transform.
        Args:
            sample: Sample dictionary to process.
            sample_idx: Index of the sample for logging purposes.
        Returns:
            Transformed sample data if successful, None otherwise.
        """
        try:
            if not self._validate_sample_structure(sample):
                self.logger.error(f"Sample {sample_idx} has invalid structure")
                return None

            result = self.transform(sample)
            self.logger.info(f"Successfully processed sample {sample_idx}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to process sample {sample_idx}: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def visualize_samples(self, info_pkl_path: str, indices: Optional[List[int]] = None) -> None:
        """
        Visualize multiple samples from info.pkl file.
        Args:
            info_pkl_path: Path to the info.pkl file.
            indices: Optional list of sample indices to process. If None, all samples are processed.
        """
        try:
            samples_list = self.load_info_pkl(info_pkl_path)

            if not samples_list:
                self.logger.warning("No samples found in info.pkl")
                return

            # Determine which samples to process
            if indices is not None:
                samples_to_process = [samples_list[i] for i in indices if i < len(samples_list)]
            else:
                samples_to_process = samples_list

            self.logger.info(f"Processing {len(samples_to_process)} samples")

            # Process each sample
            for i, sample in enumerate(samples_to_process):
                self.process_single_sample(sample, i + 1)

            self.logger.info("Finished processing all samples")

        except Exception as e:
            self.logger.error(f"Failed to visualize samples: {e}")
            self.logger.debug(traceback.format_exc())

    def visualize_single_sample(self, info_pkl_path: str, sample_idx: int) -> Optional[Dict[str, Any]]:
        """
        Visualize a single sample from info.pkl file.
        Args:
            info_pkl_path: Path to the info.pkl file.
            sample_idx: Index of the sample to process (0-based).
        Returns:
            Transformed sample data if successful, None otherwise.
        """
        try:
            samples_list = self.load_info_pkl(info_pkl_path)

            if sample_idx >= len(samples_list):
                self.logger.error(f"Sample index {sample_idx} out of range (0-{len(samples_list)-1})")
                return None

            self.logger.info(f"Processing sample {sample_idx} from {len(samples_list)} total samples")

            sample = samples_list[sample_idx]
            result = self.process_single_sample(sample, sample_idx)

            if result and self.output_dir:
                self.logger.info(f"Visualizations saved to directory: {self.output_dir}")

            return result

        except Exception as e:
            self.logger.error(f"Failed to visualize single sample: {e}")
            self.logger.debug(traceback.format_exc())
            return None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    Returns:
        Configured argument parser.
    """
    examples = """
Examples:
  # Process all samples
  python visualize_calibration_and_image.py --info_pkl data/info.pkl --data_root data/ --output_dir /vis
  # Process specific sample
  python visualize_calibration_and_image.py --info_pkl data/info.pkl --data_root data/ --output_dir /vis --sample_idx 0
  # Process specific indices
  python visualize_calibration_and_image.py --info_pkl data/info.pkl --data_root data/ --output_dir /vis --indices 0 1 2
"""

    parser = argparse.ArgumentParser(
        description="Visualize LiDAR points projected on camera images using info.pkl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )

    parser.add_argument("--info_pkl", required=True, help="Path to info.pkl file containing calibration data")
    parser.add_argument("--output_dir", help="Output directory for saving visualizations")
    parser.add_argument("--data_root", help="Root directory for data files (images, point clouds, etc.)")
    parser.add_argument("--sample_idx", type=int, help="Specific sample index to process (0-based)")
    parser.add_argument("--indices", nargs="+", type=int, help="Specific sample indices to process (0-based)")
    parser.add_argument(
        "--show_point_details", action="store_true", help="Show detailed point cloud field information"
    )

    return parser


def main() -> None:
    """
    Main entry point for the calibration visualization script.
    Parses command line arguments and runs the appropriate visualization mode.
    Supports both single sample and batch processing modes.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize visualizer
    visualizer = CalibrationVisualizer(data_root=args.data_root, output_dir=args.output_dir)

    # Run appropriate visualization mode
    if args.sample_idx is not None:
        # Process single sample
        visualizer.visualize_single_sample(args.info_pkl, args.sample_idx)
    else:
        # Process all samples or specific indices
        visualizer.visualize_samples(args.info_pkl, args.indices)


if __name__ == "__main__":
    main()
