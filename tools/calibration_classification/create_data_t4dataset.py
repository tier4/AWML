import argparse
import os
import os.path as osp
import re
from typing import Any, Dict, List, Optional, Tuple

import mmengine
import numpy as np
import yaml
from mmengine.config import Config
from mmengine.logging import MMLogger
from pyquaternion import Quaternion
from t4_devkit import Tier4
from t4_devkit.schema import Sample

# Configuration constants
DEFAULT_LIDAR_CHANNEL = "LIDAR_CONCAT"
SUPPORTED_SPLITS = ["train", "val", "test"]

# Configure logging
logger = MMLogger.get_instance(name="create_data_t4dataset")


def is_image_likely_black(img_path: str, size_threshold_bytes: int) -> bool:
    """Check if an image is likely black based on file size.

    Black images compress extremely well (nearly all zeros), resulting in very small files.
    This is much faster than reading and analyzing pixel values.

    Args:
        img_path: Path to the image file.
        size_threshold_bytes: File size threshold in bytes. Images smaller than this
            are considered likely black.

    Returns:
        True if the image file is smaller than threshold (likely black), False otherwise.
    """
    if not osp.isfile(img_path):
        logger.warning(f"Image file not found: {img_path}")
        return True

    file_size = osp.getsize(img_path)
    return file_size < size_threshold_bytes


def calculate_size_threshold(
    t4: Tier4,
    root_path: str,
    scene_root: str,
    target_cameras: List[str],
    num_samples: int = 20,
    threshold_ratio: float = 0.05,
) -> int:
    """Calculate file size threshold for filtering likely black images.

    Samples a number of images to determine average file size, then sets threshold
    as a fraction of that average. Black images are typically <5% of normal image size.

    Args:
        t4: Tier4 dataset instance.
        root_path: Absolute root path of the dataset.
        scene_root: Relative scene root path.
        target_cameras: List of camera channels to sample from.
        num_samples: Number of images to sample for calculating average size.
        threshold_ratio: Ratio of average size to use as threshold (default 0.05 = 5%).

    Returns:
        Size threshold in bytes. Images smaller than this are likely black.
    """
    image_sizes: List[int] = []

    for sd in t4.sample_data:
        if len(image_sizes) >= num_samples:
            break

        filename = sd.filename
        for cam in target_cameras:
            if filename.startswith(f"data/{cam}/"):
                img_rel_path = osp.join(scene_root, filename)
                img_abs_path = osp.join(root_path, img_rel_path)
                if osp.isfile(img_abs_path):
                    file_size = osp.getsize(img_abs_path)
                    # Only include non-tiny files in the average calculation
                    # (to avoid black images skewing the average down)
                    if file_size > 10000:  # > 10KB
                        image_sizes.append(file_size)
                break

    if not image_sizes:
        # Fallback to a reasonable default (50KB) if no valid samples found
        logger.warning("Could not sample images for threshold calculation, using default 50KB threshold")
        return 50000

    avg_size = sum(image_sizes) / len(image_sizes)
    threshold = int(avg_size * threshold_ratio)

    # Ensure minimum threshold of 10KB to avoid false positives
    threshold = max(threshold, 10000)

    logger.info(
        f"Calculated size threshold: {threshold / 1000:.1f}KB "
        f"(sampled {len(image_sizes)} images, avg size: {avg_size / 1000:.1f}KB)"
    )

    return threshold


def get_samples_excluded_by_velocity(
    t4: Tier4,
    lidar_channel: str,
    max_velocity_mps: float,
) -> set:
    """Compute which sample indices to exclude based on velocity from pose derivative.

    Velocity is estimated as (position[i] - position[i-1]) / (time[i] - time[i-1]) using
    ego pose translation and timestamp. The first sample is always excluded (no derivative).
    Samples with speed above max_velocity_mps are excluded.

    Args:
        t4: Tier4 dataset instance for the scene.
        lidar_channel: Name of the lidar channel used to get ego pose per sample.
        max_velocity_mps: Maximum acceptable speed in m/s; samples above this are excluded.

    Returns:
        Set of 0-based sample indices to exclude. Empty set if max_velocity_mps <= 0.
    """
    if max_velocity_mps <= 0:
        return set()

    excluded: set = set()
    samples = t4.sample

    if len(samples) == 0:
        return excluded

    # First sample has no previous sample for derivative; always exclude
    excluded.add(0)

    for i in range(1, len(samples)):
        sample_curr = samples[i]
        sample_prev = samples[i - 1]

        lidar_token_curr = sample_curr.data.get(lidar_channel)
        lidar_token_prev = sample_prev.data.get(lidar_channel)
        if lidar_token_curr is None or lidar_token_prev is None:
            excluded.add(i)
            continue

        sd_curr = t4.get("sample_data", lidar_token_curr)
        sd_prev = t4.get("sample_data", lidar_token_prev)
        if not sd_curr.is_valid or not sd_prev.is_valid:
            excluded.add(i)
            continue

        ep_curr = t4.get("ego_pose", sd_curr.ego_pose_token)
        ep_prev = t4.get("ego_pose", sd_prev.ego_pose_token)

        pos_curr = np.array(ep_curr.translation)
        pos_prev = np.array(ep_prev.translation)
        # Timestamps are in microseconds
        dt_sec = (ep_curr.timestamp - ep_prev.timestamp) * 1e-6
        if dt_sec <= 0:
            excluded.add(i)
            continue

        velocity = (pos_curr - pos_prev) / dt_sec
        speed_mps = float(np.linalg.norm(velocity))
        if speed_mps > max_velocity_mps:
            excluded.add(i)

    return excluded


def build_transform_matrix(rotation: Quaternion, translation: np.ndarray) -> List[List[float]]:
    """Build a 4x4 transformation matrix from rotation quaternion and translation vector.

    Args:
        rotation: Rotation as a pyquaternion Quaternion.
        translation: Translation as a numpy array [x, y, z].

    Returns:
        4x4 transformation matrix as nested lists.
    """
    mat = np.eye(4)
    mat[:3, :3] = rotation.rotation_matrix
    mat[:3, 3] = np.array(translation)
    return mat.tolist()


def calculate_lidar2cam_matrix(
    lidar2ego: Optional[List[List[float]]],
    lidar_pose: Optional[List[List[float]]],
    cam_pose: Optional[List[List[float]]],
    cam2ego: Optional[List[List[float]]],
    camera_name: str = "unknown",
) -> List[List[float]]:
    """Calculate lidar2cam transformation matrix.

    Args:
        lidar2ego: Lidar to ego vehicle transformation matrix.
        lidar_pose: Lidar pose transformation matrix.
        cam_pose: Camera pose transformation matrix.
        cam2ego: Camera to ego vehicle transformation matrix.
        camera_name: Name of the camera for logging purposes.

    Returns:
        lidar2cam transformation matrix as nested lists.

    Raises:
        ValueError: If any of the required transformation matrices are missing.
    """
    if None in (lidar2ego, lidar_pose, cam_pose, cam2ego):
        missing_matrices = []
        if lidar2ego is None:
            missing_matrices.append("lidar2ego")
        if lidar_pose is None:
            missing_matrices.append("lidar_pose")
        if cam_pose is None:
            missing_matrices.append("cam_pose")
        if cam2ego is None:
            missing_matrices.append("cam2ego")

        error_msg = f"Missing required transformation matrices for camera {camera_name}: {', '.join(missing_matrices)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        lidar2ego_mat = np.array(lidar2ego)
        lidar_pose_mat = np.array(lidar_pose)
        cam_pose_mat = np.array(cam_pose)
        cam2ego_mat = np.array(cam2ego)
        cam_pose_inv = np.linalg.inv(cam_pose_mat)
        cam2ego_inv = np.linalg.inv(cam2ego_mat)

        lidar2cam = cam2ego_inv @ cam_pose_inv @ lidar_pose_mat @ lidar2ego_mat
        return lidar2cam.tolist()
    except (ValueError, np.linalg.LinAlgError) as e:
        error_msg = f"Failed to calculate lidar2cam matrix for camera {camera_name}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error calculating lidar2cam matrix for camera {camera_name}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def extract_frame_index(filename: str) -> str:
    """Extract the frame index (all digits before the first dot in the basename).

    Args:
        filename: The filename to extract frame index from.

    Returns:
        The extracted frame index as a string.

    Raises:
        ValueError: If no valid frame index can be extracted.
    """
    if not filename:
        raise ValueError("Empty filename provided")

    base = osp.basename(filename)
    # Get the part before the first dot
    before_dot = base.split(".", 1)[0]
    # Find all digits in that part
    digits = re.findall(r"(\d+)", before_dot)
    if digits:
        # Validate that the first digit group looks like a frame index
        frame_idx = digits[0]
        if len(frame_idx) > 0 and frame_idx.isdigit():
            return frame_idx

    # If no valid digits found, try the entire before_dot part
    if before_dot and before_dot.isdigit():
        return before_dot

    raise ValueError(f"Could not extract valid frame index from filename: {filename}")


def _get_modality_value(sensor_record: Any) -> Optional[str]:
    """Safely extract modality string value from a sensor record.

    Handles both enum (SensorModality) and plain string modality fields.

    Args:
        sensor_record: A sensor record with a modality attribute.

    Returns:
        Modality as a lowercase string, or None if not available.
    """
    modality = getattr(sensor_record, "modality", None)
    if modality is None:
        return None
    return modality.value if hasattr(modality, "value") else modality


def get_available_cameras(t4: Tier4, target_cameras: Optional[List[str]] = None) -> List[str]:
    """Get available camera channels from the dataset.

    Args:
        t4: Tier4 dataset instance.
        target_cameras: Optional list of desired camera channels.
            If provided, only matching cameras are returned.
            If None, all camera channels in the dataset are returned.

    Returns:
        Sorted list of camera channel names.
    """
    all_cameras = sorted(sensor.channel for sensor in t4.sensor if _get_modality_value(sensor) == "camera")
    if target_cameras is not None:
        return [cam for cam in target_cameras if cam in all_cameras]
    return all_cameras


def get_lidar_sources_info(t4: Tier4) -> Dict[str, Dict[str, Any]]:
    """Collect all lidar sensors and their calibrated extrinsics.

    Args:
        t4: Tier4 dataset instance.

    Returns:
        Dictionary mapping channel name (e.g. LIDAR_CONCAT) to sensor info:
        - sensor_token: Sensor token from the dataset.
        - translation: [x, y, z] in meters (sensor-to-base).
        - rotation: 3x3 rotation matrix as nested list (sensor-to-base).
    """
    lidar_sources: Dict[str, Dict[str, Any]] = {}
    for cs_rec in getattr(t4, "calibrated_sensor", []):
        try:
            sensor_rec = t4.get("sensor", cs_rec.sensor_token)
        except KeyError:
            continue

        modality_value = _get_modality_value(sensor_rec)
        if modality_value != "lidar":
            continue

        channel = sensor_rec.channel
        if channel not in lidar_sources:
            rot_matrix = cs_rec.rotation.rotation_matrix
            lidar_sources[channel] = dict(
                sensor_token=sensor_rec.token,
                translation=np.array(cs_rec.translation).tolist(),
                rotation=rot_matrix.tolist(),
            )

    return lidar_sources


def build_sample_infos(
    t4: Tier4,
    sample: Sample,
    sample_idx: int,
    scene_root: str,
    target_cameras: List[str],
    lidar_channel: str,
    scene_id: str,
    lidar_sources: Dict[str, Any],
    root_path: Optional[str] = None,
    size_threshold: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build calibration info dicts for a single sample, one per camera.

    Args:
        t4: Tier4 dataset instance.
        sample: Sample (keyframe) record to process.
        sample_idx: Starting index for sample numbering.
        scene_root: Relative scene root path (from root_path).
        target_cameras: List of camera channels to process.
        lidar_channel: Name of the lidar channel.
        scene_id: ID of the scene.
        lidar_sources: Dict mapping lidar channel names to sensor info (sensor_token, translation [x,y,z], rotation 3x3 matrix). Computed once per scene.
        root_path: Absolute root path of the dataset for image validation.
        size_threshold: File size threshold in bytes for filtering likely black images.

    Returns:
        List of info dictionaries, one per available camera in this sample.
    """
    # Get lidar sample_data token
    lidar_token = sample.data.get(lidar_channel)
    if lidar_token is None:
        logger.warning(f"No lidar channel '{lidar_channel}' in sample {sample.token}, scene {scene_id}, skipping.")
        return []

    sd_lidar = t4.get("sample_data", lidar_token)
    if not sd_lidar.is_valid:
        logger.warning(f"Lidar sample_data is invalid for sample {sample.token}, scene {scene_id}, skipping.")
        return []

    cs_lidar = t4.get("calibrated_sensor", sd_lidar.calibrated_sensor_token)
    ep_lidar = t4.get("ego_pose", sd_lidar.ego_pose_token)

    lidar2ego = build_transform_matrix(cs_lidar.rotation, cs_lidar.translation)
    lidar_pose = build_transform_matrix(ep_lidar.rotation, ep_lidar.translation)
    frame_idx = extract_frame_index(sd_lidar.filename)

    lidar_data: Dict[str, Any] = {
        "lidar_path": osp.join(scene_root, sd_lidar.filename),
        "lidar_pose": lidar_pose,
        "lidar2ego": lidar2ego,
        "timestamp": sd_lidar.timestamp,
        "sample_data_token": sd_lidar.token,
    }

    infos: List[Dict[str, Any]] = []
    valid_count = 0

    for cam in target_cameras:
        cam_token = sample.data.get(cam)
        if cam_token is None:
            continue

        sd_cam = t4.get("sample_data", cam_token)
        if not sd_cam.is_valid:
            logger.debug(f"Camera {cam} sample_data is invalid in frame {frame_idx}, scene {scene_id}, skipping.")
            continue

        cs_cam = t4.get("calibrated_sensor", sd_cam.calibrated_sensor_token)
        ep_cam = t4.get("ego_pose", sd_cam.ego_pose_token)

        cam2ego = build_transform_matrix(cs_cam.rotation, cs_cam.translation)
        cam_pose = build_transform_matrix(ep_cam.rotation, ep_cam.translation)

        cam_info: Dict[str, Any] = {
            "img_path": osp.join(scene_root, sd_cam.filename),
            "cam2img": np.array(cs_cam.camera_intrinsic).tolist(),
            "cam2ego": cam2ego,
            "cam_pose": cam_pose,
            "distortion_coefficients": np.array(cs_cam.camera_distortion).tolist(),
            "sample_data_token": sd_cam.token,
            "timestamp": sd_cam.timestamp,
            "height": sd_cam.height,
            "width": sd_cam.width,
        }

        # Check if image is likely black based on file size (when filtering is enabled)
        if size_threshold is not None and root_path is not None:
            img_abs_path = osp.join(root_path, cam_info["img_path"])
            if is_image_likely_black(img_abs_path, size_threshold):
                file_size = osp.getsize(img_abs_path) if osp.isfile(img_abs_path) else 0
                logger.warning(
                    f"Skipping likely black image for camera {cam} in frame {frame_idx}, "
                    f"scene {scene_id} (size: {file_size / 1000:.1f}KB < threshold {size_threshold / 1000:.1f}KB)"
                )
                continue

        # Calculate lidar2cam transformation matrix
        try:
            cam_info["lidar2cam"] = calculate_lidar2cam_matrix(
                lidar2ego, lidar_pose, cam_pose, cam2ego, camera_name=cam
            )
        except ValueError as e:
            logger.error(f"Failed to process frame {frame_idx} for camera {cam} in scene {scene_id}: {e}")
            raise

        info: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "frame_id": cam,
            "image": cam_info,
            "lidar_points": lidar_data,
            "lidar_sources": lidar_sources,
            "sample_idx": sample_idx + valid_count,
            "scene_id": scene_id,
        }
        infos.append(info)
        valid_count += 1

    return infos


def generate_scene_calib_info(
    t4: Tier4,
    scene_root: str,
    scene_id: str,
    start_sample_idx: int = 0,
    target_cameras: Optional[List[str]] = None,
    lidar_channel: str = DEFAULT_LIDAR_CHANNEL,
    root_path: Optional[str] = None,
    filter_black_images: bool = False,
    max_velocity_mps: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Generate calibration info for all keyframe samples in a scene.

    Each info contains camera and lidar data with their respective transformation matrices,
    plus lidar sources info for the scene.

    Args:
        t4: Tier4 dataset instance for the scene.
        scene_root: Relative scene root path (from root_path).
        scene_id: ID of the scene.
        start_sample_idx: Starting index for sample numbering.
        target_cameras: List of target camera channels to process.
            If None, processes all available cameras.
        lidar_channel: Name of the lidar channel.
        root_path: Absolute root path of the dataset for image validation.
        filter_black_images: Whether to filter out black images.
        max_velocity_mps: If set and > 0, exclude samples with velocity above this (m/s),
            estimated from pose derivative. First sample per scene is always excluded.

    Returns:
        Tuple of (infos, next_sample_idx) where infos is a list of calibration
        info dictionaries and next_sample_idx is the next available sample index.
    """
    # Resolve available cameras
    if target_cameras is None:
        target_cameras = get_available_cameras(t4)
        logger.info(f"Using all available cameras: {target_cameras}")

    # Calculate size threshold for black image filtering (once per scene)
    size_threshold: Optional[int] = None
    if filter_black_images and root_path is not None:
        size_threshold = calculate_size_threshold(t4, root_path, scene_root, target_cameras)

    # Compute samples to exclude by velocity (first sample + those above threshold)
    velocity_excluded: set = set()
    if max_velocity_mps is not None and max_velocity_mps > 0:
        velocity_excluded = get_samples_excluded_by_velocity(t4, lidar_channel, max_velocity_mps)
        if velocity_excluded:
            logger.info(
                f"Velocity filter: excluding {len(velocity_excluded)} samples in scene {scene_id} "
                f"(max_velocity_mps={max_velocity_mps})"
            )

    # Compute lidar sources info (once per scene)
    lidar_sources = get_lidar_sources_info(t4)

    logger.info(f"Processing {len(t4.sample)} samples in scene {scene_id}")

    infos: List[Dict[str, Any]] = []
    sample_idx = start_sample_idx

    for i, sample in enumerate(t4.sample):
        if i in velocity_excluded:
            continue
        try:
            sample_infos = build_sample_infos(
                t4,
                sample,
                sample_idx,
                scene_root,
                target_cameras,
                lidar_channel,
                scene_id,
                lidar_sources,
                root_path,
                size_threshold,
            )
            if sample_infos:
                infos.extend(sample_infos)
                sample_idx += len(sample_infos)
        except ValueError as e:
            logger.error(f"Failed to process sample {sample.token} in scene {scene_id}: {e}")
            raise

    return infos, sample_idx


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the T4dataset calibration info creation script.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Create calibration info for T4dataset (classification version)")
    parser.add_argument("--config", type=str, required=True, help="config for T4dataset")
    parser.add_argument("--root_path", type=str, required=True, help="specify the root path of dataset")
    parser.add_argument("--version", type=str, required=True, help="product version")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="output directory of info file")
    parser.add_argument(
        "--lidar_channel", default=DEFAULT_LIDAR_CHANNEL, help=f"Lidar channel name (default: {DEFAULT_LIDAR_CHANNEL})"
    )
    parser.add_argument(
        "--target_cameras", nargs="*", default=None, help="Target cameras to generate info for (default: all cameras)"
    )
    parser.add_argument(
        "--filter-black-images", action="store_true", help="Filter out black images (all pixels are zero)"
    )
    parser.add_argument(
        "--filter-velocity",
        type=float,
        default=None,
        metavar="MPS",
        help="Filter out samples with velocity above threshold (m/s). Velocity is estimated from pose derivative; the first sample per scene is always excluded. Example: --filter-velocity 50.0",
    )
    return parser.parse_args()


def get_scene_root_dir_path(root_path: str, dataset_version: str, scene_id: str) -> str:
    """Get the scene root directory path, handling version subdirectories.

    Args:
        root_path: Root path of the dataset.
        dataset_version: Version of the dataset.
        scene_id: ID of the scene (format: "uuid/version", e.g., "c4edef0a-1d00-4f0d-9e08-65ba8b0692ff/0").

    Returns:
        Path to the scene root directory.
    """
    scene_id = scene_id.strip()
    scene_id_parts = scene_id.split("/")

    if len(scene_id_parts) == 2 and scene_id_parts[1].isdigit():
        base_scene_id = scene_id_parts[0]
        version_id = scene_id_parts[1]
        scene_root_dir_path = osp.join(root_path, dataset_version, base_scene_id)
        return osp.join(scene_root_dir_path, version_id)

    raise ValueError(f"Invalid scene_id format: {scene_id}. Expected 'uuid/version'.")


def main() -> None:
    """Main function to create calibration info for T4dataset.

    This function:
    1. Parses command line arguments
    2. Loads configuration files
    3. Iterates through dataset versions and scenes
    4. Instantiates Tier4 for each scene and generates calibration info
    5. Saves the results to pickle files for each split (train/val/test)
    """
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info(f"Starting calibration info creation with config: {args.config}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Lidar channel: {args.lidar_channel}")
    if args.target_cameras:
        logger.info(f"Target cameras: {args.target_cameras}")
    if args.filter_black_images:
        logger.info("Black image filtering is enabled")
    if args.filter_velocity is not None and args.filter_velocity > 0:
        logger.info(f"Velocity filtering is enabled (max {args.filter_velocity} m/s)")

    abs_root_path = osp.abspath(args.root_path)

    split_infos: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SUPPORTED_SPLITS}
    split_sample_idx: Dict[str, int] = {split: 0 for split in SUPPORTED_SPLITS}

    logger.info(f"Processing dataset versions: {cfg.dataset_version_list}")
    for dataset_version in cfg.dataset_version_list:
        dataset_list = osp.join(cfg.dataset_version_config_root, dataset_version + ".yaml")
        with open(dataset_list, "r") as f:
            dataset_list_dict: Dict[str, Any] = yaml.safe_load(f)

        for split in SUPPORTED_SPLITS:
            for scene_id in dataset_list_dict.get(split, []):
                scene_root_dir_path = get_scene_root_dir_path(args.root_path, dataset_version, scene_id)
                annotation_dir = osp.join(scene_root_dir_path, "annotation")

                logger.debug(
                    f"split={split}, scene_id={scene_id}, "
                    f"annotation_dir={annotation_dir}, exists={osp.isdir(annotation_dir)}"
                )

                if not osp.isdir(annotation_dir):
                    logger.warning(f"Annotation dir not found: {annotation_dir}, skip.")
                    continue

                logger.info(f"Generating calibration info for {scene_id} ({split}) ...")
                rel_scene_root = osp.relpath(scene_root_dir_path, abs_root_path)

                try:
                    t4 = Tier4(data_root=scene_root_dir_path, verbose=False)
                    scene_infos, split_sample_idx[split] = generate_scene_calib_info(
                        t4,
                        rel_scene_root,
                        scene_id,
                        split_sample_idx[split],
                        args.target_cameras,
                        args.lidar_channel,
                        abs_root_path,
                        args.filter_black_images,
                        args.filter_velocity,
                    )
                    split_infos[split].extend(scene_infos)
                except ValueError as e:
                    logger.error(f"Failed to process scene {scene_id} ({split}): {e}")
                    raise

    # Save per split
    metainfo: Dict[str, str] = dict(version=args.version)
    logger.info("Saving processed data to pickle files...")
    for split in SUPPORTED_SPLITS:
        out_path = osp.join(args.out_dir, f"t4dataset_{args.version}_infos_{split}.pkl")
        mmengine.dump(dict(data_list=split_infos[split], metainfo=metainfo), out_path)
        logger.info(f"Saved {len(split_infos[split])} samples to {out_path}")

    total_samples: int = sum(len(split_infos[split]) for split in SUPPORTED_SPLITS)
    logger.info(f"Calibration info creation completed. Total samples processed: {total_samples}")


if __name__ == "__main__":
    main()
