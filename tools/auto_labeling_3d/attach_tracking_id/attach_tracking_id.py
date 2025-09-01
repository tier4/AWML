import argparse
import collections
import copy
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from attrs import define
from tqdm import tqdm

from tools.auto_labeling_3d.attach_tracking_id.mot import KalmanBoxTracker, MOTModel
from tools.auto_labeling_3d.utils.logger import setup_logger


@define
class SceneBoundary:
    """
    Represents the boundary (start and end frame) of a scene in the dataset.

    Attributes:
        scene_id (str): Unique identifier for the scene.
        scene_start_frame (int): Index of the first frame in the scene.
        scene_end_frame (int): Index of the last frame in the scene.
    """

    scene_id: str
    scene_start_frame: int
    scene_end_frame: int


def determine_scene_range(dataset_info: Dict[str, Any]):
    """
    Determine the start and end frame indices for each scene in the dataset.

    Args:
        dataset_info (Dict[str, Any]): Dictionary containing dataset information, including a 'data_list' key with frame data.

    Returns:
        Values of a dictionary mapping scene_id to SceneBoundary, representing the start and end frame indices for each scene.
    """
    scene_frames = collections.defaultdict(list)
    for frame_id, frame_info in enumerate(dataset_info["data_list"]):
        scene_id: str = frame_info["lidar_points"]["lidar_path"].split("/")[-4]
        scene_frames[scene_id].append(frame_id)

    scene_boundaries = []
    for scene_id, frames in scene_frames.items():
        scene_boundaries.append(
            SceneBoundary(
                scene_id=scene_id,
                scene_start_frame=min(frames),
                scene_end_frame=max(frames),
            )
        )
    return scene_boundaries


def track_objects(
    dataset_info: Dict[str, Any],
    scene_boundary: SceneBoundary,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Update instance id by using object tracking on the dataset.

    Apply multi-object tracking to frames between start_frame and end_frame
    in the specified scene, assigning unique IDs to each object.

    Args:
        dataset_info: Dictionary containing dataset information
            - metainfo: Meta information (class names, etc.)
            - data_list: List of frame data
        scene_boundary: SceneBoundary object containing scene_id, scene_start_frame, and scene_end_frame.
        logger: Logger instance for output messages

    Returns:
        Dict[str, Any]: Dataset information with tracking results
            Returns the input dataset_info with instance_id_3d added
            to each object in the frames
    """
    mot_model = MOTModel(classes=dataset_info["metainfo"]["classes"])

    logger.info(f"Start tracking in {scene_boundary.scene_id}")

    for frame_info in tqdm(
        dataset_info["data_list"][scene_boundary.scene_start_frame : scene_boundary.scene_end_frame + 1]
    ):
        ego2global = np.array(frame_info["ego2global"])

        tracked_instance_ids = mot_model.frame_mot(
            frame_info["pred_instances_3d"], ego2global, frame_info["timestamp"]
        )
        # update instance_id by tracking
        for det_index, tracked_instance_id in enumerate(tracked_instance_ids):
            frame_info["pred_instances_3d"][det_index]["instance_id_3d"] = tracked_instance_id

    logger.info(f"Total number of tracks in {scene_boundary.scene_id}: {KalmanBoxTracker.count}")
    return dataset_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track objects using Kalman filter and Hungarian algorithm with BEV IoU"
    )

    parser.add_argument("--input", type=str, required=True, help="Path to input pseudo labeled pkl file")
    parser.add_argument("--output", type=str, required=True, help="Path to output tracked pkl file")
    parser.add_argument(
        "--log-level",
        help="Set log level",
        default="INFO",
        choices=list(logging._nameToLevel.keys()),
    )
    parser.add_argument("--work-dir", help="the directory to save logs")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger: logging.Logger = setup_logger(args, name="attach_tracking_id")

    # load info
    with open(input_path, "rb") as f:
        dataset_info = pickle.load(f)

    scene_boundaries = determine_scene_range(dataset_info)
    for scene_boundary in scene_boundaries:
        dataset_info = track_objects(dataset_info, scene_boundary, logger)

    # save tracked info
    with open(output_path, "wb") as f:
        pickle.dump(dataset_info, f)
    logger.info(f"Tracked dataset saved to {output_path}")


if __name__ == "__main__":
    main()
