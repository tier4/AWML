import argparse
import json
import os
import os.path as osp
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import mmengine
import numpy as np
import yaml
from mmengine.config import Config
from mmengine.logging import MMLogger
from pyquaternion import Quaternion

# Configuration constants
DEFAULT_LIDAR_CHANNEL = "LIDAR_CONCAT"
SUPPORTED_SPLITS = ["train", "val", "test"]
REQUIRED_JSON_FILES = ["sample_data.json", "ego_pose.json", "calibrated_sensor.json"]

# Configure logging
logger = MMLogger.get_instance(name="create_data_t4dataset")


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file from the given path.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        PermissionError: If the file cannot be read due to permissions.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {path}: {e}", e.doc, e.pos)
    except PermissionError:
        raise PermissionError(f"Cannot read file due to permissions: {path}")


def convert_quaternion_to_matrix(rotation: List[float], translation: List[float]) -> List[List[float]]:
    """Convert quaternion rotation and translation to a 4x4 transformation matrix.
    Args:
        rotation (list): Quaternion [w, x, y, z].
        translation (list): Translation [x, y, z].
    Returns:
        list: 4x4 transformation matrix as nested lists.
    """

    rot = Quaternion(rotation).rotation_matrix
    trans = np.array(translation).reshape(3, 1)
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3:4] = trans
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


def get_pose_dict(ego_pose_json: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create a dict mapping pose token to pose entry from ego_pose_json.
    Args:
        ego_pose_json (list): List of ego pose dictionaries from the annotation file.
    Returns:
        dict: Dictionary mapping pose tokens to their corresponding pose entries.
    """
    return {ego_pose["token"]: ego_pose for ego_pose in ego_pose_json}


def get_calib_dict(calib_json: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create a dict mapping calibration token to calibration entry from calib_json.
    Args:
        calib_json (list): List of calibrated sensor dictionaries from the annotation file.
    Returns:
        dict: Dictionary mapping calibration tokens to their corresponding calibration entries.
    """
    return {calib["token"]: calib for calib in calib_json}


def get_all_channels(sample_data_json: List[Dict[str, Any]]) -> List[str]:
    """Return all unique camera channels by parsing filename in sample_data_json.
    Args:
        sample_data_json (list): List of sample data dictionaries from the annotation file.
    Returns:
        list: Sorted list of unique camera channel names (e.g., ['CAM_FRONT', 'CAM_LEFT', ...]).
    """
    return sorted(
        set(
            sample_data["filename"].split("/")[1]
            for sample_data in sample_data_json
            if sample_data["filename"].startswith("data/CAM_")
        )
    )


def extract_frame_index(filename: str) -> str:
    """
    Extract the frame index (all digits before the first dot in the basename).
    E.g. 00001.jpg -> 00001, 123.pcd.bin -> 123, 5.jpg -> 5
    Args:
        filename (str): The filename to extract frame index from.
    Returns:
        str: The extracted frame index as a string.
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


def generate_calib_info(
    annotation_dir: str,
    lidar_channel: str = DEFAULT_LIDAR_CHANNEL,
    scene_root: Optional[str] = None,
    scene_id: Optional[str] = None,
    start_sample_idx: int = 0,
    target_cameras: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Generate calibration info for each frame (grouped by filename index).
    Each info contains all sensor sample_data (lidar, cameras, etc) belonging to the same frame.
    Invalid samples (is_valid = false) are automatically filtered out.
    Args:
        annotation_dir (str): Path to the annotation directory containing JSON files.
        lidar_channel (str, optional): Name of the lidar channel. Defaults to "LIDAR_CONCAT".
        scene_root (str, optional): Root path of the scene data. Defaults to None.
        scene_id (str, optional): ID of the scene. Defaults to None.
        start_sample_idx (int, optional): Starting index for sample numbering. Defaults to 0.
        target_cameras (list, optional): List of target camera channels to process.
            If None, processes all available cameras. Defaults to None.
    Returns:
        tuple: (infos, next_sample_idx) where infos is a list of calibration info dictionaries
               and next_sample_idx is the next available sample index.
    """
    # Validate required files exist
    for required_file in REQUIRED_JSON_FILES:
        file_path = osp.join(annotation_dir, required_file)
        if not osp.isfile(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    sample_data_json = load_json(osp.join(annotation_dir, "sample_data.json"))
    ego_pose_json = load_json(osp.join(annotation_dir, "ego_pose.json"))
    calib_json = load_json(osp.join(annotation_dir, "calibrated_sensor.json"))
    calib_dict = get_calib_dict(calib_json)
    ego_pose_dict = get_pose_dict(ego_pose_json)

    # Get all available camera channels if target_cameras is not specified
    if target_cameras is None:
        target_cameras = get_all_channels(sample_data_json)
        logger.info(f"Using all available cameras: {target_cameras}")

    # Group all sample_data by frame index, filtering out invalid samples
    frame_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    invalid_count = 0
    for sd in sample_data_json:
        if "filename" not in sd or not sd["filename"]:
            continue
        # Skip invalid sample data (is_valid = false)
        if sd.get("is_valid") is False:
            invalid_count += 1
            continue
        frame_idx: str = extract_frame_index(sd["filename"])
        frame_groups[frame_idx].append(sd)

    if invalid_count > 0:
        logger.warning(
            f"Found {len(frame_groups)} frames in scene {scene_id} (filtered out {invalid_count} invalid samples)"
        )
    else:
        logger.info(f"Found {len(frame_groups)} frames in scene {scene_id}")

    infos: List[Dict[str, Any]] = []
    sample_idx: int = start_sample_idx
    for _, (frame_idx, frame_sample_data) in enumerate(sorted(frame_groups.items())):
        try:
            frame_infos = build_frame_info(
                frame_idx,
                frame_sample_data,
                calib_dict,
                ego_pose_dict,
                scene_root,
                target_cameras,
                lidar_channel,
                scene_id,
                sample_idx,
            )
            if frame_infos:
                infos.extend(frame_infos)
                sample_idx += len(frame_infos)
        except ValueError as e:
            logger.error(f"Failed to process frame {frame_idx} in scene {scene_id}: {e}")
            raise
    return infos, sample_idx


def build_frame_info(
    frame_idx: str,
    frame_sample_data: List[Dict[str, Any]],
    calib_dict: Dict[str, Dict[str, Any]],
    ego_pose_dict: Dict[str, Dict[str, Any]],
    scene_root: str,
    target_cameras: List[str],
    lidar_channel: str,
    scene_id: Optional[str] = None,
    sample_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Build frame info for each target camera. If target_cameras is None, build for all cameras.
    Args:
        frame_idx (str): Frame index identifier.
        frame_sample_data (list): List of sample data dictionaries for this frame.
        calib_dict (dict): Dictionary mapping calibration tokens to calibration entries.
        ego_pose_dict (dict): Dictionary mapping pose tokens to pose entries.
        scene_root (str): Root path of the scene data.
        target_cameras (list): List of target camera channels to process.
        lidar_channel (str): Name of the lidar channel.
        scene_id (str, optional): ID of the scene. Defaults to None.
        sample_idx (int, optional): Starting index for sample numbering. Defaults to 0.
    Returns:
        list: List of frame info dictionaries, one for each available camera in the frame.
              Each info contains camera data, lidar data, and transformation matrices.
    Raises:
        ValueError: If no lidar data is found for the given frame.
    """

    # Find lidar data for this frame
    lidar_data: Optional[Dict[str, Any]] = None
    for sd in frame_sample_data:
        filename = sd["filename"]
        if filename.startswith(f"data/{lidar_channel}/"):
            lidar_calib = calib_dict[sd["calibrated_sensor_token"]]
            lidar_pose = ego_pose_dict[sd["ego_pose_token"]]
            lidar_data = {
                "lidar_path": osp.join(scene_root, filename),
                "lidar_pose": convert_quaternion_to_matrix(lidar_pose["rotation"], lidar_pose["translation"]),
                "lidar2ego": convert_quaternion_to_matrix(lidar_calib["rotation"], lidar_calib["translation"]),
                "timestamp": sd["timestamp"],
                "sample_data_token": sd["token"],
            }
            break

    if lidar_data is None:
        logger.warning(f"No lidar data found for frame {frame_idx} in scene {scene_id}, skipping frame.")
        return []

    # Find camera data for this frame
    camera_data: Dict[str, Dict[str, Any]] = {}
    for sd in frame_sample_data:
        filename = sd["filename"]
        for cam in target_cameras:
            if filename.startswith(f"data/{cam}/"):
                cam_calib = calib_dict[sd["calibrated_sensor_token"]]
                cam_pose = ego_pose_dict[sd["ego_pose_token"]]
                camera_data[cam] = {
                    "img_path": osp.join(scene_root, filename),
                    "cam2img": cam_calib["camera_intrinsic"],
                    "cam2ego": convert_quaternion_to_matrix(cam_calib["rotation"], cam_calib["translation"]),
                    "cam_pose": convert_quaternion_to_matrix(cam_pose["rotation"], cam_pose["translation"]),
                    "distortion_coefficients": cam_calib["camera_distortion"],
                    "sample_data_token": sd["token"],
                    "timestamp": sd["timestamp"],
                    "height": sd["height"],
                    "width": sd["width"],
                }
                break

    # Filter to only include cameras that exist in the data
    available_cameras: List[str] = [cam for cam in target_cameras if cam in camera_data]

    infos: List[Dict[str, Any]] = []
    for i, cam in enumerate(available_cameras):
        cam_info = camera_data[cam]

        # Calculate lidar2cam transformation matrix
        lidar2ego = lidar_data["lidar2ego"]
        lidar_pose = lidar_data["lidar_pose"]
        cam_pose = cam_info["cam_pose"]
        cam2ego = cam_info["cam2ego"]

        try:
            cam_info["lidar2cam"] = calculate_lidar2cam_matrix(
                lidar2ego, lidar_pose, cam_pose, cam2ego, camera_name=cam
            )
        except ValueError as e:
            logger.error(f"Failed to process frame {frame_idx} for camera {cam} in scene {scene_id}: {e}")
            raise

        # Create info for this camera
        info = {
            "frame_idx": frame_idx,
            "frame_id": cam,
            "image": cam_info,
            "lidar_points": lidar_data,
            "sample_idx": sample_idx + i,
            "scene_id": scene_id,
        }
        infos.append(info)

    return infos


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the T4dataset calibration info creation script.
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - config: Path to the T4dataset configuration file
            - root_path: Root path of the dataset
            - version: Product version
            - out_dir: Output directory for info files
            - lidar_channel: Lidar channel name (default: LIDAR_CONCAT)
            - target_cameras: List of target cameras to process (default: all cameras)
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
    return parser.parse_args()


def get_scene_root_dir_path(root_path: str, dataset_version: str, scene_id: str) -> str:
    """Get the scene root directory path, handling version subdirectories.
    Args:
        root_path (str): Root path of the dataset.
        dataset_version (str): Version of the dataset.
        scene_id (str): ID of the scene (may include version number like "uuid version").
    Returns:
        str: Path to the scene root directory. If scene_id contains a version number,
             returns the path with the version. Otherwise, looks for version subdirectories
             and returns the path to the highest version number subdirectory.
    """
    # Check if scene_id already contains a version number (e.g., "uuid version")
    scene_id_parts = scene_id.strip().split()
    if len(scene_id_parts) == 2 and scene_id_parts[1].isdigit():
        # scene_id contains version number, use it directly
        base_scene_id = scene_id_parts[0]
        version_id = scene_id_parts[1]
        scene_root_dir_path = osp.join(root_path, dataset_version, base_scene_id)
        return osp.join(scene_root_dir_path, version_id)
    else:
        raise ValueError(f"Invalid scene_id format: {scene_id}")


def main() -> None:
    """Main function to create calibration info for T4dataset.
    This function:
    1. Parses command line arguments
    2. Loads configuration files
    3. Iterates through dataset versions and scenes
    4. Generates calibration info for each scene
    5. Saves the results to pickle files for each split (train/val/test)
    The script processes T4dataset annotations to create calibration information
    that includes camera and lidar data with their respective transformation matrices.
    """
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info(f"Starting calibration info creation with config: {args.config}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Lidar channel: {args.lidar_channel}")
    if args.target_cameras:
        logger.info(f"Target cameras: {args.target_cameras}")

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
                scene_root = get_scene_root_dir_path(args.root_path, dataset_version, scene_id)
                annotation_dir = osp.join(scene_root, "annotation")
                logger.debug(
                    f"split={split}, scene_id={scene_id}, annotation_dir={annotation_dir}, exists={osp.isdir(annotation_dir)}"
                )
                if not osp.isdir(annotation_dir):
                    logger.warning(f"Annotation dir not found: {annotation_dir}, skip.")
                    continue
                logger.info(f"Generating calibration info for {scene_id} ({split}) ...")
                rel_scene_root = osp.relpath(scene_root, abs_root_path)
                scene_infos: List[Dict[str, Any]]
                try:
                    scene_infos, split_sample_idx[split] = generate_calib_info(
                        annotation_dir,
                        args.lidar_channel,
                        rel_scene_root,
                        scene_id,
                        split_sample_idx[split],
                        args.target_cameras,
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
