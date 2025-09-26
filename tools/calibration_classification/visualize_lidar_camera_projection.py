#!/usr/bin/env python3

import argparse
import os
import pickle
import traceback
from typing import Any, Dict, List, Optional, Union

import numpy as np
from mmengine.config import Config
from mmengine.logging import MMLogger

from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)


class CalibrationToolkit:
    """
    A comprehensive tool for processing LiDAR-camera calibration data.
    This class provides functionality to load calibration data from info.pkl files
    and optionally generate visualizations or save results as NPZ files.
    Attributes:
        transform: The calibration classification transform instance
        data_root: Root directory for data files
        output_dir: Directory for saving visualizations (only used if visualize=True)
        logger: MMLogger instance for logging
        collected_results: List to store processed results for NPZ saving
    """

    def __init__(self, model_cfg: Config, data_root: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize the CalibrationToolkit.
        Args:
            model_cfg: Model configuration
            data_root: Root directory for data files. If None, absolute paths are used.
            output_dir: Directory for saving visualizations. If None, no visualizations are saved.
        """
        self.model_cfg = model_cfg
        self.data_root = data_root
        self.output_dir = output_dir
        self.transform = None
        self.logger = MMLogger.get_instance(name="calibration_toolkit")
        self.collected_results = []
        self._initialize_transform()

    def _initialize_transform(self) -> None:
        """Initialize the CalibrationClassificationTransform with appropriate parameters."""
        transform_config = self.model_cfg.get("transform_config", None)
        if transform_config is None:
            raise ValueError("transform_config not found in model configuration")

        # Only set projection_vis_dir if output_dir is provided (for visualization)
        projection_vis_dir = self.output_dir if self.output_dir else None

        self.transform = CalibrationClassificationTransform(
            transform_config=transform_config,
            mode="test",
            undistort=True,
            data_root=self.data_root,
            projection_vis_dir=projection_vis_dir,
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

    def process_single_sample(
        self, sample: Dict[str, Any], sample_idx: int, save_npz: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single sample using the transform.
        Args:
            sample: Sample dictionary to process.
            sample_idx: Index of the sample for logging purposes.
            save_npz: Whether to collect result for NPZ saving.
        Returns:
            Transformed sample data if successful, None otherwise.
        """
        try:
            if not self._validate_sample_structure(sample):
                self.logger.error(f"Sample {sample_idx} has invalid structure")
                return None

            result = self.transform(sample)

            # Store the result for NPZ saving only if requested
            if save_npz:
                self.collected_results.append(result["fused_img"])

            self.logger.info(f"Successfully processed sample {sample_idx}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to process sample {sample_idx}: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def save_npz_file(self, output_path: str) -> None:
        """
        Save all collected results as an NPZ file with the correct structure.
        Args:
            output_path: Path where to save the NPZ file.
        """
        if not self.collected_results:
            self.logger.warning("No results collected to save as NPZ")
            return

        try:
            # Convert list of arrays to a single array with shape (number_of_samples, 5, 1860, 2880)
            # Each result['fused_img'] has shape (1860, 2880, 5), so we need to transpose
            input_array = np.array([result.transpose(2, 0, 1) for result in self.collected_results], dtype=np.float32)

            # Save as NPZ file
            np.savez(output_path, input=input_array)

            self.logger.info(f"Saved NPZ file with shape {input_array.shape} to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save NPZ file: {e}")
            self.logger.debug(traceback.format_exc())

    def process_samples(
        self,
        info_pkl_path: str,
        indices: Optional[List[int]] = None,
        visualize: bool = False,
        save_npz: bool = False,
        npz_output_path: Optional[str] = None,
    ) -> None:
        """
        Process multiple samples from info.pkl file.
        Args:
            info_pkl_path: Path to the info.pkl file.
            indices: Optional list of sample indices to process. If None, all samples are processed.
            visualize: Whether to generate visualizations (requires output_dir to be set).
            save_npz: Whether to collect results for NPZ saving.
            npz_output_path: Path for saving NPZ file (only used if save_npz=True).
        """
        try:
            samples_list = self.load_info_pkl(info_pkl_path)

            if not samples_list:
                self.logger.warning("No samples found in info.pkl")
                return

            # Clear previous results
            self.collected_results = []

            # Determine which samples to process
            if indices is not None:
                samples_to_process = [samples_list[i] for i in indices if i < len(samples_list)]
            else:
                samples_to_process = samples_list

            self.logger.info(f"Processing {len(samples_to_process)} samples")
            if visualize:
                self.logger.info("Visualization enabled")
            if save_npz:
                self.logger.info("NPZ saving enabled")

            # Process each sample
            for i, sample in enumerate(samples_to_process):
                self.process_single_sample(sample, i + 1, save_npz=save_npz)

            # Save NPZ file if requested
            if save_npz and npz_output_path:
                self.save_npz_file(npz_output_path)

            self.logger.info("Finished processing all samples")

        except Exception as e:
            self.logger.error(f"Failed to process samples: {e}")
            self.logger.debug(traceback.format_exc())

    def process_single_sample_from_file(
        self,
        info_pkl_path: str,
        sample_idx: int,
        visualize: bool = False,
        save_npz: bool = False,
        npz_output_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single sample from info.pkl file.
        Args:
            info_pkl_path: Path to the info.pkl file.
            sample_idx: Index of the sample to process (0-based).
            visualize: Whether to generate visualizations (requires output_dir to be set).
            save_npz: Whether to collect results for NPZ saving.
            npz_output_path: Path for saving NPZ file (only used if save_npz=True).
        Returns:
            Transformed sample data if successful, None otherwise.
        """
        try:
            samples_list = self.load_info_pkl(info_pkl_path)

            if sample_idx >= len(samples_list):
                self.logger.error(f"Sample index {sample_idx} out of range (0-{len(samples_list)-1})")
                return None

            self.logger.info(f"Processing sample {sample_idx} from {len(samples_list)} total samples")
            if visualize:
                self.logger.info("Visualization enabled")
            if save_npz:
                self.logger.info("NPZ saving enabled")

            # Clear previous results for single sample processing
            self.collected_results = []

            sample = samples_list[sample_idx]
            result = self.process_single_sample(sample, sample_idx, save_npz=save_npz)

            if result and visualize and self.output_dir:
                self.logger.info(f"Visualizations saved to directory: {self.output_dir}")

            # Save NPZ file if requested
            if save_npz and npz_output_path and result:
                self.save_npz_file(npz_output_path)
            return result

        except Exception as e:
            self.logger.error(f"Failed to process single sample: {e}")
            self.logger.debug(traceback.format_exc())
            return None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    Returns:
        Configured argument parser.
    """
    examples = """
Examples:
  # Process all samples without visualization or NPZ saving
  python toolkit.py model_config.py --info_pkl data/info.pkl --data_root data/
  # Process all samples with visualization
  python toolkit.py model_config.py --info_pkl data/info.pkl --data_root data/ --output_dir /vis --visualize
  # Process all samples and save as NPZ
  python toolkit.py model_config.py --info_pkl data/info.pkl --data_root data/ --npz_output_path results.npz --save_npz
  # Process all samples with both visualization and NPZ saving
  python toolkit.py model_config.py --info_pkl data/info.pkl --data_root data/ --output_dir /vis --visualize --npz_output_path results.npz --save_npz
  # Process specific sample with visualization
  python toolkit.py model_config.py --info_pkl data/info.pkl --data_root data/ --output_dir /vis --visualize --sample_idx 0
  # Process specific indices with NPZ saving
  python toolkit.py model_config.py --info_pkl data/info.pkl --data_root data/ --save_npz --npz_output_path results.npz --indices 0 1 2
  # Process first 5 samples (indices 0, 1, 2, 3, 4) with both features
  python toolkit.py model_config.py --info_pkl data/info.pkl --data_root data/ --output_dir /vis --visualize --save_npz --npz_output_path results.npz --indices 5
"""

    parser = argparse.ArgumentParser(
        description="Process LiDAR-camera calibration data with optional visualization and NPZ saving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )

    parser.add_argument("model_cfg", help="model config path")
    parser.add_argument("--info_pkl", required=True, help="Path to info.pkl file containing calibration data")
    parser.add_argument("--output_dir", help="Output directory for saving visualizations (only used with --visualize)")
    parser.add_argument("--data_root", help="Root directory for data files (images, point clouds, etc.)")
    parser.add_argument("--sample_idx", type=int, help="Specific sample index to process (0-based)")
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        help="Specific sample indices to process (0-based), or a single number N to process indices 0 to N-1",
    )
    parser.add_argument("--visualize", action="store_true", help="Enable visualization (requires --output_dir)")
    parser.add_argument("--save_npz", action="store_true", help="Enable NPZ saving (requires --npz_output_path)")
    parser.add_argument("--npz_output_path", help="Path for saving NPZ file (only used with --save_npz)")
    parser.add_argument(
        "--show_point_details", action="store_true", help="Show detailed point cloud field information"
    )

    return parser


def main() -> None:
    """
    Main entry point for the calibration toolkit script.
    Parses command line arguments and runs the appropriate processing mode.
    Supports both single sample and batch processing modes with optional features.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate argument combinations
    if args.visualize and not args.output_dir:
        parser.error("--visualize requires --output_dir to be specified")

    if args.save_npz and not args.npz_output_path:
        parser.error("--save_npz requires --npz_output_path to be specified")

    # Load model configuration
    model_cfg = Config.fromfile(args.model_cfg)

    # Initialize toolkit
    toolkit = CalibrationToolkit(model_cfg=model_cfg, data_root=args.data_root, output_dir=args.output_dir)

    # Process indices argument
    processed_indices = None
    if args.indices is not None:
        if len(args.indices) == 1:
            # If only one number provided, treat it as range 0 to N-1
            n = args.indices[0]
            processed_indices = list(range(n))
            toolkit.logger.info(f"Processing indices 0 to {n-1} (total: {n} samples)")
        else:
            # If multiple numbers provided, use them as specific indices
            processed_indices = args.indices
            toolkit.logger.info(f"Processing specific indices: {processed_indices}")

    # Run appropriate processing mode
    if args.sample_idx is not None:
        # Process single sample
        toolkit.process_single_sample_from_file(
            args.info_pkl,
            args.sample_idx,
            visualize=args.visualize,
            save_npz=args.save_npz,
            npz_output_path=args.npz_output_path,
        )
    else:
        # Process all samples or specific indices
        toolkit.process_samples(
            args.info_pkl,
            processed_indices,
            visualize=args.visualize,
            save_npz=args.save_npz,
            npz_output_path=args.npz_output_path,
        )


if __name__ == "__main__":
    main()
