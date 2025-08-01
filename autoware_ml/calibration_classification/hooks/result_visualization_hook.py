import os
from typing import List, Optional

import cv2
import numpy as np
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.registry import HOOKS

from autoware_ml.calibration_classification.datasets.transforms.calibration_classification_transform import (
    CalibrationClassificationTransform,
)


@HOOKS.register_module()
class ResultVisualizationHook(Hook):
    """
    Hook for visualizing the results of calibration classification during validation and testing.
    This hook saves visualizations of the original and undistorted images, along with prediction results, to a specified directory.
    Note: Training phase is not supported for result visualization (only projection visualization is available).
    Args:
        data_root (str, optional): Root directory for dataset images. Used to resolve relative image paths.
        results_vis_dir (str, optional): Directory to save results visualization.
        phases (list, optional): List of phases to enable visualization for.
                               Options: ['val', 'test']. Default: ['test']
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        results_vis_dir: Optional[str] = None,
        phases: Optional[List[str]] = None,
    ):
        """
        Initialize the ResultVisualizationHook.
        Args:
            data_root (str, optional): Root directory for dataset images.
            results_vis_dir (str, optional): Directory to save results visualization.
            phases (list, optional): List of phases to enable visualization for.
                                   Options: ['val', 'test']. Default: ['test']
                                   Note: Training phase is not supported for result visualization.
        """
        self.transform = CalibrationClassificationTransform(data_root=data_root, results_vis_dir=results_vis_dir)
        self.data_root = data_root
        self.results_vis_dir = results_vis_dir
        self.phases = phases if phases is not None else ["test"]
        self.logger = MMLogger.get_current_instance()

    def after_val_iter(self, runner, batch_idx: int, data_batch=None, outputs: Optional[List] = None):
        """
        Called after each validation iteration to visualize and save results.
        Args:
            runner: The runner object handling the validation loop.
            batch_idx (int): Index of the current batch.
            data_batch: The input data batch (not used).
            outputs (list): List of DataSample objects, one per batch element.
        """
        if "val" in self.phases:
            self._process_iteration_outputs(outputs, "val")

    def after_test_iter(self, runner, batch_idx: int, data_batch=None, outputs: Optional[List] = None):
        """
        Called after each test iteration to visualize and save results.
        Args:
            runner: The runner object handling the test loop.
            batch_idx (int): Index of the current batch.
            data_batch: The input data batch (not used).
            outputs (list): List of DataSample objects, one per batch element.
        """
        if "test" in self.phases:
            self._process_iteration_outputs(outputs, "test")

    def _process_iteration_outputs(self, outputs: Optional[List], phase: str):
        """
        Process outputs from any iteration phase (val/test).
        Args:
            outputs (list): List of DataSample objects, one per batch element.
            phase (str): Current phase ('val' or 'test').
        """
        if outputs is None:
            return

        for output in outputs:
            self._process_single_output(output, phase)

    def _process_single_output(self, output, phase: str):
        """Process a single output for visualization."""
        try:
            # Check if output is a valid DataSample object
            if not hasattr(output, "pred_label") or not hasattr(output, "metainfo"):
                self.logger.warning(
                    f"[ResultVisualizationHook] Invalid output type in {phase} phase: {type(output)}. Expected DataSample object."
                )
                return

            # Extract prediction label
            pred_label = self._extract_prediction_label(output)

            # Get image path and validate
            img_path = self._get_image_path(output)
            if not img_path:
                return

            # Load and validate original image
            original_image = self._load_original_image(img_path)
            if original_image is None:
                return

            # Create undistorted image
            undistorted_image = self._create_undistorted_image(output, original_image)

            # Visualize results
            self._visualize_results(output, pred_label, original_image, undistorted_image, img_path, phase)

        except Exception as e:
            self.logger.error(f"[ResultVisualizationHook] Error processing output in {phase} phase: {e}")

    def _extract_prediction_label(self, output) -> int:
        """Extract prediction label from output."""
        pred_label = output.pred_label
        if hasattr(pred_label, "item"):
            return int(pred_label.item())
        else:
            return int(pred_label)

    def _get_image_path(self, output) -> Optional[str]:
        """Get and validate image path from output metadata."""
        if not hasattr(output, "metainfo") or not isinstance(output.metainfo, dict):
            self.logger.warning(
                "[ResultVisualizationHook] metainfo not found or not a dictionary, skipping visualization."
            )
            return None

        if "image" not in output.metainfo:
            self.logger.warning("[ResultVisualizationHook] image not found in metainfo, skipping visualization.")
            return None

        img_path = output.metainfo["image"].get("img_path")
        if not img_path:
            self.logger.warning("[ResultVisualizationHook] img_path not found for image, skipping visualization.")
            return None

        return img_path

    def _load_original_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load and validate original image from path."""
        # Resolve full image path
        img_path_full = self._resolve_image_path(img_path)

        if not os.path.exists(img_path_full):
            self.logger.warning(f"[ResultVisualizationHook] img_path does not exist on disk: {img_path_full}")
            return None

        original_image = cv2.imread(img_path_full)
        if original_image is None:
            self.logger.warning(f"[ResultVisualizationHook] cv2.imread failed to load image: {img_path_full}")
            return None

        return original_image

    def _resolve_image_path(self, img_path: str) -> str:
        """Resolve relative image path to absolute path."""
        if not os.path.isabs(img_path) and getattr(self.transform, "data_root", None):
            return os.path.join(self.transform.data_root, img_path)
        return img_path

    def _create_undistorted_image(self, output, original_image: np.ndarray) -> np.ndarray:
        """Create undistorted image using camera calibration parameters."""
        try:
            cam_info = output.metainfo["image"]
            camera_matrix = np.array(cam_info["cam2img"])
            distortion_coefficients = np.zeros(5, dtype=np.float32)  # Use zeros if not available

            return cv2.undistort(
                original_image,
                camera_matrix,
                distortion_coefficients,
                newCameraMatrix=camera_matrix,
            )
        except (KeyError, TypeError, ValueError) as e:
            self.logger.warning(
                f"[ResultVisualizationHook] Failed to create undistorted image: {e}. Using original image."
            )
            return original_image

    def _visualize_results(
        self,
        output,
        pred_label: int,
        original_image: np.ndarray,
        undistorted_image: np.ndarray,
        img_path: str,
        phase: str,
    ):
        """Visualize and save results."""
        try:
            input_data = output.metainfo.get("fused_img", None)
            frame_id = output.metainfo.get("frame_id", None)
            if input_data is None:
                self.logger.warning(
                    f"[ResultVisualizationHook] fused_img data not found for {img_path}, skipping visualization."
                )
                return

            # Extract image index and sample index
            frame_idx = output.metainfo.get("frame_idx", None)
            sample_idx = output.metainfo.get("sample_idx", 0)  # Default to 0 if not available

            # Perform visualization with phase information
            self.transform.visualize_results(
                input_data,
                pred_label,
                original_image,
                undistorted_image,
                frame_idx=frame_idx,
                sample_idx=sample_idx,
                phase=phase,
                frame_id=frame_id,
            )
        except Exception as e:
            self.logger.error(f"[ResultVisualizationHook] Error in visualization for {img_path}: {e}")
