import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from mmcv.transforms import BaseTransform
from mmengine.logging import MMLogger
from mmpretrain.registry import TRANSFORMS

from autoware_ml.calibration_classification.datasets.transforms.camera_lidar_augmentation import alter_calibration

logger = MMLogger.get_instance(name="calibration_classification_transform")


class TransformMode(Enum):
    """Enumeration for transform modes."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class CalibrationData:
    """Structured representation of camera calibration data.
    This class holds all the necessary calibration information for camera-LiDAR
    coordinate transformations and image processing operations.
    """

    camera_matrix: npt.NDArray[np.float32]  # Original camera intrinsic matrix (3x3)
    distortion_coefficients: npt.NDArray[np.float32]  # Camera distortion coefficients
    lidar_to_camera_transformation: npt.NDArray[np.float32]
    # Updated camera matrix after image processing (undistortion, cropping, scaling)
    # This matrix should be used for 3D->2D projection after any image transformations
    # to ensure geometric consistency between the processed image and 3D point projections
    new_camera_matrix: Optional[npt.NDArray[np.float32]] = None

    def __post_init__(self):
        """Initialize new_camera_matrix if not provided.
        Sets new_camera_matrix to a copy of camera_matrix if it's None.
        """
        if self.new_camera_matrix is None:
            self.new_camera_matrix = self.camera_matrix.copy()


@TRANSFORMS.register_module()
class CalibrationClassificationTransform(BaseTransform):
    """Transforms image, pointcloud, and calibration data into inputs for a calibration classifier.
    This transform processes camera images, LiDAR point clouds, and calibration data
    to generate inputs suitable for training a calibration classification model.
    It supports image undistortion, data augmentation, and LiDAR projection.
    """

    def __init__(
        self,
        transform_config: Dict[str, Any],
        mode: str = "train",
        undistort: bool = True,
        enable_augmentation: bool = True,
        data_root: str = None,
        projection_vis_dir: Optional[str] = None,
        results_vis_dir: Optional[str] = None,
    ):
        """Initialize the CalibrationClassificationTransform.
        Args:
            transform_config (Dict[str, Any]): Configuration for transform parameters. Must contain:
                - crop_ratio: Crop ratio for image augmentation (0.0, 1.0]
                - max_distortion: Maximum distortion for affine transformation [0.0, 1.0]
                - depth_scale: Scale factor for depth visualization (> 0.0)
                - radius: Radius for point visualization (>= 0)
                - rgb_weight: Weight for RGB overlay [0.0, 1.0]
            mode (str): Transform mode. Options: "train", "val", "test". Defaults to "train".
            undistort (bool): Whether to undistort images. Defaults to True.
            enable_augmentation (bool): Whether to enable data augmentation. Defaults to True.
            data_root (str): Root path for data files. Defaults to None.
            projection_vis_dir (Optional[str]): Directory to save projection visualization results. Defaults to None.
            results_vis_dir (Optional[str]): Directory to save results visualization results. Defaults to None.
        Raises:
            ValueError: If mode is invalid, data_root doesn't exist, or transform_config is invalid.
        """
        super().__init__()

        # Validate mode parameter
        try:
            self.mode = TransformMode(mode)
        except ValueError:
            valid_modes = [m.value for m in TransformMode]
            raise ValueError(f"Invalid mode: '{mode}'. Must be one of {valid_modes}")

        # Validate data_root if provided
        if data_root is not None and not os.path.exists(data_root):
            raise ValueError(f"Data root does not exist: {data_root}")

        self.undistort = undistort
        self.enable_augmentation = enable_augmentation
        self.transform_config = transform_config
        self.data_root = data_root
        self.projection_vis_dir = projection_vis_dir
        self.results_vis_dir = results_vis_dir
        self._validate_config()
        logger.info(f"Initialized CalibrationClassificationTransform with mode: {self.mode.value}")
        logger.info(f"Transform config: {self.transform_config}")

    def _validate_config(self) -> None:
        """Validate configuration parameters.
        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate transform_config is not None and contains required keys
        if self.transform_config is None:
            raise ValueError("transform_config cannot be None")

        required_keys = ["crop_ratio", "max_distortion", "depth_scale", "radius", "rgb_weight"]

        missing_keys = [key for key in required_keys if key not in self.transform_config]
        if missing_keys:
            raise ValueError(f"transform_config is missing required keys: {missing_keys}")

        if not isinstance(self.undistort, bool):
            raise ValueError(f"undistort must be a boolean, got {type(self.undistort)}")

        if not isinstance(self.enable_augmentation, bool):
            raise ValueError(f"enable_augmentation must be a boolean, got {type(self.enable_augmentation)}")

        if self.data_root is not None and not os.path.exists(self.data_root):
            raise ValueError(f"Data root does not exist: {self.data_root}")

        # Validate augmentation parameters
        if not (0.0 < self.transform_config["crop_ratio"] <= 1.0):
            raise ValueError(f"crop_ratio must be in (0.0, 1.0], got {self.transform_config['crop_ratio']}")

        if not (0.0 <= self.transform_config["max_distortion"] <= 1.0):
            raise ValueError(f"max_distortion must be in [0.0, 1.0], got {self.transform_config['max_distortion']}")

        if self.transform_config["depth_scale"] <= 0.0:
            raise ValueError(f"depth_scale must be positive, got {self.transform_config['depth_scale']}")

        if self.transform_config["radius"] < 0:
            raise ValueError(f"radius must be non-negative, got {self.transform_config['radius']}")

        if not (0.0 <= self.transform_config["rgb_weight"] <= 1.0):
            raise ValueError(f"rgb_weight must be in [0.0, 1.0], got {self.transform_config['rgb_weight']}")

        logger.debug("Configuration validation passed")

    @property
    def is_train(self) -> bool:
        """Check if current mode is training."""
        return self.mode == TransformMode.TRAIN

    @property
    def is_val(self) -> bool:
        """Check if current mode is validation."""
        return self.mode == TransformMode.VAL

    @property
    def is_test(self) -> bool:
        """Check if current mode is test."""
        return self.mode == TransformMode.TEST

    @property
    def is_augmentation_enabled(self) -> bool:
        """Check if augmentation is enabled for current mode."""
        return self.enable_augmentation and self.is_train

    def transform(self, results: Dict[str, Any], force_generate_miscalibration: bool = False) -> Dict[str, Any]:
        """Transform input data for calibration classification.
        Args:
            results (Dict[str, Any]): Input data dictionary containing all info.pkl fields.
            force_generate_miscalibration (bool): Whether to force generation of miscalibration. Defaults to False.
        Returns:
            Dict[str, Any]: Transformed data dictionary with processed images and labels.
        """
        logger.debug(f"Starting transform for sample {results.get('sample_idx', 'unknown')}")

        # Load and process data
        camera_data, lidar_data, calibration_data = self._load_data(results)
        logger.debug(
            f"Loaded data: camera shape {camera_data.shape}, lidar points {lidar_data['pointcloud'].shape[0]}"
        )

        undistorted_data = self._process_image(camera_data, calibration_data)
        label = self._generate_label(force_generate_miscalibration, calibration_data)
        logger.debug(f"Generated label: {label} (0=miscalibrated, 1=correct)")

        # Apply augmentations
        augmented_image, augmented_calibration, augmentation_tf = self._apply_augmentations(
            undistorted_data, calibration_data
        )

        # Generate input data
        input_data = self._generate_input_data(augmented_image, lidar_data, augmented_calibration, augmentation_tf)
        logger.debug(f"Generated input data shape: {input_data.shape}")

        # Visualization
        if self.projection_vis_dir is not None:
            self._visualize_projection(input_data, label, results, self.mode.value)

        results["fused_img"] = input_data
        results["gt_label"] = label
        logger.debug(f"Transform completed for sample {results.get('sample_idx', 'unknown')}")
        return results

    def _load_data(
        self, sample: dict
    ) -> Tuple[npt.NDArray[np.uint8], Dict[str, npt.NDArray[np.float32]], CalibrationData]:
        """Load camera, LiDAR, and calibration data from a sample dict.
        Args:
            sample: Sample dictionary from info.pkl.
        Returns:
            Tuple of camera image, LiDAR data dictionary, and calibration data.
        """
        image = self._load_image(sample)

        lidar_data = self._load_lidar_data(sample)
        calibration_data = self._load_calibration_data(sample)

        return image, lidar_data, calibration_data

    def _load_image(self, sample: dict) -> npt.NDArray[np.uint8]:
        """Load and validate camera image.
        Args:
            sample: Sample dictionary containing image information.
        Returns:
            Loaded camera image as numpy array in BGR format.
        Raises:
            FileNotFoundError: If image file is not found.
            ValueError: If image loading fails.
        """
        if "image" not in sample:
            raise KeyError("Sample does not contain 'image' key")

        img_path = sample["image"]["img_path"]

        if img_path is not None and self.data_root is not None:
            img_path = os.path.join(self.data_root, img_path)

        if img_path is None or not os.path.exists(img_path):
            logger.error(f"Image file not found: {img_path}")
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = cv2.imread(img_path)
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            raise ValueError(f"Failed to load image: {img_path}")

        logger.debug(f"Successfully loaded image from {img_path}, shape: {image.shape}")
        return image

    def _load_lidar_data(self, sample: dict) -> Dict[str, npt.NDArray[np.float32]]:
        """Load and process LiDAR data.
        Args:
            sample: Sample dictionary containing LiDAR information.
        Returns:
            Dictionary containing pointcloud and intensities.
        Raises:
            FileNotFoundError: If LiDAR file is not found.
        """
        lidar_path = sample["lidar_points"]["lidar_path"]
        if lidar_path is not None and self.data_root is not None:
            lidar_path = os.path.join(self.data_root, lidar_path)

        if lidar_path is None or not os.path.exists(lidar_path):
            logger.error(f"Lidar file not found: {lidar_path}")
            raise FileNotFoundError(f"Lidar file not found: {lidar_path}")

        num_pts_feats = sample["lidar_points"].get("num_pts_feats", 5)
        pointcloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, num_pts_feats)

        logger.debug(
            f"Successfully loaded LiDAR data from {lidar_path}, points: {pointcloud.shape[0]}, features: {pointcloud.shape[1]}"
        )

        return {
            "pointcloud": pointcloud[:, :3],
            "intensities": pointcloud[:, 3],
        }

    def _load_calibration_data(self, sample: dict) -> CalibrationData:
        """Load and validate calibration data.
        Args:
            sample: Sample dictionary containing calibration information.
        Returns:
            CalibrationData object with all transformation matrices.
        Raises:
            ValueError: If required calibration fields are missing.
            KeyError: If image key is not found in sample.
        """
        if "image" not in sample:
            raise KeyError("Sample does not contain 'image' key")

        if "lidar_points" not in sample:
            raise KeyError("Sample does not contain 'lidar_points' key")

        cam_info = sample["image"]

        # Validate camera
        camera_matrix = cam_info.get("cam2img", None)
        if camera_matrix is None:
            raise ValueError(f"Camera matrix (cam2img) is missing")
        camera_matrix = np.array(camera_matrix)

        if camera_matrix.shape != (3, 3):
            raise ValueError(f"Camera matrix must be 3x3, got shape {camera_matrix.shape}")

        # Validate Lidar
        lidar_to_camera_transformation = cam_info.get("lidar2cam", None)
        if lidar_to_camera_transformation is None:
            raise ValueError(f"lidar_to_camera_transformation is missing")

        lidar_to_camera_transformation = np.array(lidar_to_camera_transformation)
        if lidar_to_camera_transformation.shape != (4, 4):
            raise ValueError(
                f"lidar_to_camera_transformation must be 4x4, got shape {lidar_to_camera_transformation.shape}"
            )

        # Validate distortion coefficients
        distortion_coefficients = cam_info.get("distortion_coefficients", None)
        if distortion_coefficients is None:
            raise ValueError(f"distortion_coefficients is missing")
        distortion_coefficients = np.array(distortion_coefficients)
        if distortion_coefficients.shape != (5,):
            raise ValueError(f"distortion_coefficients must be 5, got shape {distortion_coefficients.shape}")

        return CalibrationData(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            lidar_to_camera_transformation=lidar_to_camera_transformation,
        )

    def _process_image(self, image: npt.NDArray[np.uint8], calibration_data: CalibrationData) -> npt.NDArray[np.uint8]:
        """Process image with undistortion if enabled.
        Args:
            image: Input camera image.
            calibration_data: Camera calibration parameters.
        Returns:
            Processed image (undistorted if enabled).
        """
        if self.undistort:
            image, _ = self._undistort_image(calibration_data, image)
        return image

    def _undistort_image(
        self, calibration_data: CalibrationData, image: npt.NDArray[np.uint8], alpha: float = 0.0
    ) -> Tuple[npt.NDArray[np.uint8], CalibrationData]:
        """Undistort image and update calibration data.
        Args:
            calibration_data: Camera calibration parameters.
            image: Input image to undistort.
            alpha: Free scaling parameter for undistortion.
        Returns:
            Tuple of undistorted image and updated calibration data.
        """
        if not np.any(calibration_data.distortion_coefficients):
            return image, calibration_data

        h, w = image.shape[:2]
        calibration_data.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            calibration_data.camera_matrix, calibration_data.distortion_coefficients, (w, h), alpha, (w, h)
        )
        image = cv2.undistort(
            image,
            calibration_data.camera_matrix,
            calibration_data.distortion_coefficients,
            newCameraMatrix=calibration_data.new_camera_matrix,
        )
        calibration_data.distortion_coefficients = np.zeros_like(calibration_data.distortion_coefficients)
        return image, calibration_data

    def _generate_label(self, force_generate_miscalibration: bool, calibration_data: CalibrationData) -> int:
        """Generate classification label and apply miscalibration if needed.
        Args:
            force_generate_miscalibration: Whether to force miscalibration generation.
            calibration_data: Camera calibration parameters.
        Returns:
            Classification label (0 for miscalibrated, 1 for correct).
        Raises:
            ValueError: If force_generate_miscalibration is used in test mode.
        """
        if self.is_test and force_generate_miscalibration:
            raise ValueError("force_generate_miscalibration is not supported in test mode")

        if self.is_test:
            generate_miscalibration = False
        else:
            generate_miscalibration = force_generate_miscalibration or random.choice([True, False])

        if generate_miscalibration:
            logger.debug("Generating miscalibration for training sample")
            calibration_data.lidar_to_camera_transformation = alter_calibration(
                calibration_data.lidar_to_camera_transformation,
                min_augmentation_angle=1.0,
                max_augmentation_angle=10.0,
                min_augmentation_radius=0.05,
                max_augmentation_radius=0.2,
            )
            return 0  # Miscalibrated
        else:
            logger.debug("Using correct calibration for training sample")
            return 1  # Correctly calibrated

    def _apply_augmentations(
        self, image: npt.NDArray[np.uint8], calibration_data: CalibrationData
    ) -> Tuple[npt.NDArray[np.uint8], CalibrationData, Optional[npt.NDArray[np.float32]]]:
        """Apply data augmentations if enabled.
        Args:
            image: Input image to augment.
            calibration_data: Camera calibration parameters.
        Returns:
            Tuple of augmented image, updated calibration data, and optional transformation matrix.
        """
        return (
            self._apply_augmentation_transforms(image, calibration_data)
            if self.is_augmentation_enabled
            else (image, calibration_data, None)
        )

    def _apply_augmentation_transforms(
        self, image: npt.NDArray[np.uint8], calibration_data: CalibrationData
    ) -> Tuple[npt.NDArray[np.uint8], CalibrationData, Optional[npt.NDArray[np.float32]]]:
        """Apply scaling, cropping, and affine transformations.
        Args:
            image: Input image to augment.
            calibration_data: Camera calibration parameters.
        Returns:
            Tuple of augmented image, updated calibration data, and optional transformation matrix.
        """
        # Scaling and cropping
        if random.random() > 0.5:
            logger.debug("Applying scale and crop augmentation")
            image, calibration_data = self._scale_and_crop_image(image, calibration_data)

        # Affine transformation
        affine_matrix = None
        if random.random() > 0.5:
            logger.debug("Applying affine transformation augmentation")
            image, affine_matrix = self._apply_affine_transformation(image)

        return image, calibration_data, affine_matrix

    def _scale_and_crop_image(
        self, image: npt.NDArray[np.uint8], calibration_data: CalibrationData
    ) -> Tuple[npt.NDArray[np.uint8], CalibrationData]:
        """Scale and crop image, updating camera matrix accordingly.
        Args:
            image: Input image to scale and crop.
            calibration_data: Camera calibration parameters.
        Returns:
            Tuple of scaled and cropped image with updated calibration data.
        """
        h, w = image.shape[:2]

        crop_ratio = self.transform_config["crop_ratio"]
        # Random crop center with noise
        crop_center_noise = [self._signed_random(0, crop_ratio / 2), self._signed_random(0, crop_ratio / 2)]
        crop_center = np.array([h * (1 + crop_center_noise[0]) / 2, w * (1 + crop_center_noise[1]) / 2])

        # Determine scaled dimensions
        scale_noise = np.random.uniform(crop_ratio, 1 - np.max(np.abs(crop_center_noise)))
        scaled_h, scaled_w = h * scale_noise, w * scale_noise

        # Calculate crop region
        start_h, end_h = int(crop_center[0] - scaled_h / 2), int(crop_center[0] + scaled_h / 2)
        start_w, end_w = int(crop_center[1] - scaled_w / 2), int(crop_center[1] + scaled_w / 2)

        # Enforce bounds
        start_h, end_h = max(0, start_h), min(h, end_h)
        start_w, end_w = max(0, start_w), min(w, end_w)

        # Crop and resize
        cropped_image = image[start_h:end_h, start_w:end_w]
        resized_image = cv2.resize(cropped_image, (w, h))

        # Update camera matrix
        self._update_camera_matrix_for_crop(calibration_data, start_w, start_h, end_w, end_h, w)

        return resized_image, calibration_data

    def _update_camera_matrix_for_crop(
        self, calibration_data: CalibrationData, start_w: int, start_h: int, end_w: int, end_h: int, w: int
    ) -> None:
        """Update camera matrix to account for cropping and scaling.
        Args:
            calibration_data: Camera calibration parameters to update.
            start_w: Starting width coordinate of crop.
            start_h: Starting height coordinate of crop.
            end_w: Ending width coordinate of crop.
            end_h: Ending height coordinate of crop.
            w: Original image width.
        """
        scale_factor = w / (end_w - start_w)
        calibration_data.new_camera_matrix[0, 0] *= scale_factor  # fx
        calibration_data.new_camera_matrix[1, 1] *= scale_factor  # fy

        # Update principal point coordinates
        calibration_data.new_camera_matrix[0, 2] = (calibration_data.new_camera_matrix[0, 2] - start_w) * scale_factor
        calibration_data.new_camera_matrix[1, 2] = (calibration_data.new_camera_matrix[1, 2] - start_h) * scale_factor

    def _signed_random(self, min_value: float, max_value: float) -> float:
        """Generate random value with random sign.
        Args:
            min_value: Minimum absolute value.
            max_value: Maximum absolute value.
        Returns:
            Random value with random sign.
        """
        sign = 1 if random.random() < 0.5 else -1
        return sign * random.uniform(min_value, max_value)

    def _apply_affine_transformation(
        self, image: npt.NDArray[np.uint8]
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        """
        Applies a controlled affine transformation to the image and updates the calibration matrix.
        Args:
            image (np.ndarray): Input image.
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Transformed image and the 3x3 affine transformation matrix.
        """
        h, w = image.shape[:2]
        max_distortion = self.transform_config["max_distortion"]

        # Limit distortions to a fraction of image dimensions
        max_offset_x = max_distortion * w
        max_offset_y = max_distortion * h

        # Source points (corners of the image)
        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])

        # Destination points with bounded distortion
        dst_pts = src_pts + np.random.uniform(
            low=[[-max_offset_x, -max_offset_y], [-max_offset_x, -max_offset_y], [-max_offset_x, -max_offset_y]],
            high=[[max_offset_x, max_offset_y], [max_offset_x, max_offset_y], [max_offset_x, max_offset_y]],
        ).astype(np.float32)

        # Compute and apply affine transformation
        affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)
        transformed_image = cv2.warpAffine(image, affine_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT)

        # Create 3x3 transformation matrix
        affine_transform_3x3 = np.eye(3)
        affine_transform_3x3[:2, :3] = affine_matrix

        return transformed_image, affine_transform_3x3

    def _generate_input_data(
        self,
        image: npt.NDArray[np.uint8],
        lidar_data: Dict[str, npt.NDArray[np.float32]],
        calibration_data: CalibrationData,
        augmentation_tf: Optional[npt.NDArray[np.float32]] = None,
    ) -> npt.NDArray[np.uint8]:
        """Generate depth and intensity images using augmented calibration.
        Args:
            image: Input camera image.
            lidar_data: LiDAR point cloud and intensity data.
            calibration_data: Camera calibration parameters.
            augmentation_tf: Optional augmentation transformation matrix.
        Returns:
            Combined image with RGB, depth, and intensity channels.
        """
        points = lidar_data["pointcloud"]
        num_points = points.shape[0]
        points_hom = np.concatenate([points, np.ones((num_points, 1), dtype=points.dtype)], axis=1)

        # Transform points to camera coordinate system
        pointcloud_ccs = self._transform_points_to_camera(points_hom, calibration_data)

        # Filter valid points and project to image
        valid_points = pointcloud_ccs[:, 2] > 0.0
        pointcloud_ccs = pointcloud_ccs[valid_points]
        intensities = lidar_data["intensities"][valid_points]

        # Project to image coordinates
        pointcloud_ics = self._project_points_to_image(pointcloud_ccs, calibration_data)

        # Apply augmentation transformation
        if augmentation_tf is not None:
            pointcloud_ics = self._apply_augmentation_to_points(pointcloud_ics, augmentation_tf)

        return self._create_lidar_images(image, pointcloud_ics, pointcloud_ccs, intensities)

    def _transform_points_to_camera(
        self,
        points_hom: npt.NDArray[np.float32],
        calibration_data: CalibrationData,
    ) -> npt.NDArray[np.float32]:
        """Transform points to camera coordinate system.
        Args:
            points_hom: Homogeneous point coordinates (N, 4).
            calibration_data: Camera calibration parameters.
        Returns:
            Points in camera coordinate system (N, 3).
        """

        lidar2cam = calibration_data.lidar_to_camera_transformation
        points_hom = (lidar2cam @ points_hom.T).T

        # The lidar2cam is calcuated from multi-step transformation
        # Step 1: LiDAR to baselink
        # points_hom = (lidar_to_ego @ points_hom.T).T

        # # Step 2: baselink (LiDAR time) to global
        # points_hom = (lidar_pose @ points_hom.T).T

        # # Step 3: global to baselink (Camera time)
        # points_hom = (camera_pose_inv @ points_hom.T).T

        # # Step 4: baselink to camera
        # points_hom = (camera_to_ego_inv @ points_hom.T).T

        return points_hom[:, :3]

    def _project_points_to_image(
        self, pointcloud_ccs: npt.NDArray[np.float32], calibration_data: CalibrationData
    ) -> npt.NDArray[np.float32]:
        """Project 3D points to image coordinates.
        Args:
            pointcloud_ccs: Points in camera coordinate system (N, 3).
            calibration_data: Camera calibration parameters.
        Returns:
            Points in image coordinate system (N, 2).
        """
        camera_matrix = calibration_data.new_camera_matrix
        distortion_coefficients = calibration_data.distortion_coefficients
        pointcloud_ics, _ = cv2.projectPoints(
            pointcloud_ccs, np.zeros(3), np.zeros(3), camera_matrix, distortion_coefficients
        )
        return pointcloud_ics.reshape(-1, 2)

    def _apply_augmentation_to_points(
        self, pointcloud_ics: npt.NDArray[np.float32], augmentation_tf: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Apply augmentation transformation to 2D points.
        Args:
            pointcloud_ics: Points in image coordinate system (N, 2).
            augmentation_tf: 3x3 augmentation transformation matrix.
        Returns:
            Transformed points in image coordinate system (N, 2).
        """
        num_points = pointcloud_ics.shape[0]
        homogeneous_ics = np.hstack([pointcloud_ics, np.ones((num_points, 1))])
        transformed_ics = (augmentation_tf @ homogeneous_ics.T).T[:, :2]
        return transformed_ics

    def _create_lidar_images(
        self,
        image: npt.NDArray[np.uint8],
        pointcloud_ics: npt.NDArray[np.float32],
        pointcloud_ccs: npt.NDArray[np.float32],
        intensities: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        """Create depth and intensity images.
        Args:
            image: Base camera image.
            pointcloud_ics: Point cloud in image coordinate system.
            pointcloud_ccs: Point cloud in camera coordinate system.
            intensities: LiDAR intensity values.
        Returns:
            Combined image with BGR, depth, and intensity channels.
        """
        h, w = image.shape[:2]
        depth_image = np.zeros((h, w), dtype=np.uint8)
        intensity_image = np.zeros((h, w), dtype=np.uint8)

        depth_scale = self.transform_config["depth_scale"]
        radius = self.transform_config["radius"]

        for point3d, intensity, point2d in zip(pointcloud_ccs, intensities, pointcloud_ics):
            if np.any(np.abs(point2d) > (2**31 - 1)):
                continue

            distance_color = int(np.clip(255 * point3d[2] / depth_scale, 0, 255))
            cv2.circle(depth_image, tuple(point2d.astype(int)), radius, distance_color, -1)
            cv2.circle(intensity_image, tuple(point2d.astype(int)), radius, int(intensity), -1)

        depth_image = np.expand_dims(depth_image, axis=2)
        intensity_image = np.expand_dims(intensity_image, axis=2)
        return np.concatenate([image, depth_image, intensity_image], axis=2)

    def _create_overlay_image(
        self, bgr_image: npt.NDArray[np.uint8], feature_image: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Create colored overlay image.
        Args:
            bgr_image: Base BGR image.
            feature_image: Feature image to overlay.
        Returns:
            Overlaid image with features visualized in BGR format.
        """
        rgb_weight = self.transform_config["rgb_weight"]

        overlay_image = bgr_image.copy()
        intensity_colormap = cv2.applyColorMap((feature_image).astype(np.uint8), cv2.COLORMAP_JET)
        intensity_mask = (feature_image > 0).astype(np.uint8).squeeze(-1)
        masked_colormap = intensity_colormap * intensity_mask[:, :, None]

        overlay_image[intensity_mask > 0] = cv2.addWeighted(
            overlay_image[intensity_mask > 0], rgb_weight, masked_colormap[intensity_mask > 0], 1 - rgb_weight, 0
        )
        return overlay_image

    def _visualize_projection(
        self, input_data: npt.NDArray[np.uint8], label: int, results: Dict[str, Any], phase: str = "test"
    ) -> None:
        """Visualize LiDAR projection results.
        Args:
            input_data: Combined input data with BGR, depth, and intensity channels.
            label: Classification label (0 for miscalibrated, 1 for correct).
            results: Input data dictionary.
            phase: Current phase ('train', 'val', or 'test'). Defaults to "test".
        """
        frame_id = results.get("frame_id", "unknown")
        frame_idx = results.get("frame_idx", "unknown")
        sample_idx = results["sample_idx"]

        camera_data = input_data[:, :, :3]
        intensity_image = input_data[:, :, 4:5]
        overlay_image = self._create_overlay_image(camera_data, intensity_image)
        os.makedirs(self.projection_vis_dir, exist_ok=True)
        frame_id_str = frame_id if frame_id is not None else "unknown"
        save_path = os.path.join(
            self.projection_vis_dir,
            f"projection_{phase}_sample_{sample_idx}_{frame_idx}_{frame_id_str}_gt_label_{label}.png",
        )
        cv2.imwrite(save_path, overlay_image)
        logger.info(f"Saved {phase} projection visualization to {save_path}")

    def visualize_results(
        self,
        input_data: npt.NDArray[np.uint8],
        pred_label: int,
        gt_label: int,
        original_image: npt.NDArray[np.uint8],
        undistorted_image: npt.NDArray[np.uint8],
        frame_idx: str = None,
        sample_idx: int = None,
        phase: str = "test",
        frame_id: str = None,
    ) -> None:
        """Visualize comprehensive results including all image types.
        Args:
            input_data: Combined input data with BGR, depth, and intensity channels.
            label: Classification label (0 for miscalibrated, 1 for correct).
            original_image: Original camera image in BGR format.
            undistorted_image: Undistorted camera image in BGR format.
            img_index: Unique index for filename.
            sample_idx: Sample id to include in filename.
            phase: Current phase ('train', 'val', or 'test'). Defaults to "test".
            frame_id: Camera frame id to include in filename.
        """
        if self.results_vis_dir is None:
            return
        camera_data = input_data[:, :, :3]  # BGR format
        depth_image = input_data[:, :, 3:4]
        intensity_image = input_data[:, :, 4:5]
        overlay_image = self._create_overlay_image(camera_data, intensity_image)  # Returns BGR format
        # Determine if prediction matches ground truth
        prediction_correct = pred_label == gt_label

        # Create title based on prediction and correctness
        if pred_label == 1:
            if prediction_correct:
                title = "Sensors Calibrated (Correct Prediction)"
            else:
                title = "Sensors Calibrated (Wrong Prediction)"
        else:
            if prediction_correct:
                title = "Sensors Miscalibrated (Correct Prediction)"
            else:
                title = "Sensors Miscalibrated (Wrong Prediction)"

        fig, axes = plt.subplots(2, 3, figsize=(10, 8))
        fig.suptitle(title)
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original RGB Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Undistorted RGB Image")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(cv2.cvtColor(camera_data, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Augmented RGB Image")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("LiDAR Overlay on Image")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(depth_image[:, :, 0], cmap="jet")
        axes[1, 1].set_title("Depth Image")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(intensity_image[:, :, 0], cmap="jet")
        axes[1, 2].set_title("Intensity Image")
        axes[1, 2].axis("off")

        plt.tight_layout()
        os.makedirs(self.results_vis_dir, exist_ok=True)
        frame_id_str = frame_id if frame_id is not None else "unknown"
        save_path = os.path.join(
            self.results_vis_dir,
            f"results_{phase}_sample_{sample_idx}_{frame_idx}_{frame_id_str}_pred_label_{pred_label}_gt_label_{gt_label}.png",
        )
        plt.savefig(save_path)
        logger.info(f"Saved {phase} results visualization to {save_path}")
        plt.close()
