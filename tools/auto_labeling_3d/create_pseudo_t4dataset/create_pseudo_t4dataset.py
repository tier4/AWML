import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from numpy.typing import NDArray
from pyquaternion import Quaternion
from t4_devkit.dataclass import Box3D as T4Box3D
from t4_devkit.dataclass import SemanticLabel, Shape, ShapeType

from tools.auto_labeling_3d.create_pseudo_t4dataset.pseudo_label_generator_3d import PseudoLabelGenerator3D
from tools.auto_labeling_3d.utils.logger import setup_logger


def _transform_pred_instance_to_global_t4box(
    bbox3d: List[float],
    velocity: List[float],
    confidence: float,
    label: str,
    instance_id: str,
    ego2global: NDArray,
    timestamp: float,
) -> T4Box3D:
    """Convert a detection instance to T4Box3D format and transform to global coordinates.

    Args:
        bbox3d (List[float]): 3D bounding box parameters [x, y, z, l, w, h, yaw]
            - x, y, z: Center position in ego vehicle coordinates
            - l (length): Size along x-axis in object coordinates
            - w (width): Size along y-axis in object coordinates
            - h (height): Size along z-axis in object coordinates
            - yaw: Rotation angle around z-axis in radians
        velocity (List[float]): Object velocity vector [vx, vy] in ego coordinates
        confidence (float): Detection confidence confidence [0-1]
        label (str): Object class label. e.g, "bus"
        instance_id (str): Instance ID of the object. e.g, "fade3eb7-77b4-420f-8248-b532800388a3"
        ego2global (NDArray): 4x4 transformation matrix from ego to global coordinates
            - 3x3 rotation matrix in top-left
            - Translation vector in fourth column
            - Last row is [0, 0, 0, 1]
        timestamp (float): unix timestamp. e.g, 1711672980.049259

    Returns:
        T4Box3D: 3D bounding box in T4Box3D format, transformed to global coordinates

    Example:
        >>> # bbox parameters in ego coordinates [x, y, z, l, w, h, yaw]
        >>> bbox3d = [-7.587669372558594, 5.918113708496094, 0.3056640625, 11.3125, 2.66796875, 3.30078125, -0.07647705078125]
        >>>
        >>> # velocity vector [vx, vy]
        >>> velocity = [3.708984375, -0.304443359375]
        >>>
        >>> # detection confidence and class label
        >>> confidence = 0.5257580280303955
        >>> label = "bus"
        >>>
        >>> # detection confidence and class label
        >>> instance_id = "fade3eb7-77b4-420f-8248-b532800388a3"
        >>> timestamp = 1711672980.049259
        >>>
        >>> # transformation matrix from ego to global coordinates
        >>> ego2global = np.array([
        ...     [-7.52366364e-01,  6.58483088e-01, -1.85688175e-02,  2.67971660e+04],
        ...     [-6.58744812e-01, -7.52079487e-01,  2.07780898e-02,  2.95396055e+04],
        ...     [-2.83205271e-04,  2.78648492e-02,  9.99611676e-01,  4.27038288e+00],
        ...     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ... ])
        >>>
        >>> # convert to T4Box3D and transform to global coordinates
        >>> box = _transform_pred_instance_to_global_t4box(bbox3d, velocity, confidence, label, instance_id, ego2global, timestamp)

    Note:
        - Input box must be in ego vehicle coordinates
        - Object dimensions are defined in object coordinates:
            * Length (l): Size along object's x-axis
            * Width (w): Size along object's y-axis
            * Height (h): Size along object's z-axis
        - Vertical velocity (vz) is automatically set to 0.0
        - T4Box3D is transformed to global coordinates using ego2global matrix
    """
    # [x, y, z]
    position: List[float] = bbox3d[:3]
    # quaternion
    rotation = Quaternion(axis=[0, 0, 1], radians=bbox3d[6])
    # [w, l, h]
    shape = Shape(shape_type=ShapeType.BOUNDING_BOX, size=(bbox3d[4], bbox3d[3], bbox3d[5]))
    # [vx, vy, vz]
    velocity: Tuple[float] = (*velocity, np.float64(0.0))

    box = T4Box3D(
        unix_time=int(timestamp),
        frame_id="base_link",
        semantic_label=SemanticLabel(label),
        position=position,
        rotation=rotation,
        shape=shape,
        velocity=velocity,
        confidence=confidence,
        uuid=instance_id,
    )

    # Transform box to global coord system
    box.rotate(Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
    box.translate(ego2global[:3, 3])
    box.frame_id = "map"

    return box


def _get_scene_and_frame_num(lidar_path: str) -> Tuple[str, int]:
    """Extract scene_id and frame_num from lidar path.

    The function expects a dataset structure without a dataset_version directory:
    ```
    data/t4dataset/
    └── dataset_name/
        └── scene_id/             # Extracted as scene_id
            ├── annotation/
            └── data/
                └── LIDAR_CONCAT/
                    └── 00000.pcd.bin   # Extracted as frame_num
    ```

    Args:
        lidar_path (str): Path to the lidar file (e.g., "data/t4dataset/dataset_name/scene_id/data/LIDAR_CONCAT/00000.pcd.bin")

    Returns:
        Tuple[str, int]: A tuple containing:
             scene_id (str): Name of the scene. e.g, "scene_0"
             frame_num (int): Frame id. e.g, 10

    Example:
        >>> lidar_path = "data/t4dataset/my_dataset/scene_0/data/LIDAR_CONCAT/00010.pcd.bin"
        >>> scene_id, frame_num = _get_scene_and_frame_num(lidar_path)
        >>> print(scene_id, frame_num)
        "scene_0" 10
    """
    scene_id: str = lidar_path.split("/")[-4]

    # get frame_num from lidar_path. e.g, '00010.pcd.bin' -> 10
    frame_num = int(lidar_path.split("/")[-1].split(".")[0])

    return scene_id, frame_num


def create_pseudo_t4dataset(
    pseudo_labeled_info_path: Path,
    non_annotated_dataset_path: Path,
    t4dataset_config: Dict[str, Any],
    overwrite: bool,
    logger: logging.Logger,
) -> None:
    """Create T4Dataset format dataset from pseudo-labeled data.

    Args:
        pseudo_labeled_info_path (Path): Path to pseudo-labeled info file(.pkl). e.g. ./data/t4dataset/info/pseudo_infos_raw_bevfusion.pkl
        non_annotated_dataset_path (Path): Path to non annotated dataset path. e.g, "./data/t4dataset/pseudo_xx1/"
        t4dataset_config (Dict[str, Any]): Config for generating T4dataset.
        overwrite (bool): If True, this code can overwrite sample_annotation.json even if t4dataset in non_annotated_dataset_path already have the annotation information.
        logger (logging.Logger): Logger instance for output messages.

    Returns:
        None: Results are saved to "annotation" directory of each scene included in non_annotated_dataset_path.
    """
    # load pseudo labeled info
    with open(pseudo_labeled_info_path, "rb") as f:
        pseudo_labeled_dataset_info = pickle.load(f)

    # get scene_ids
    assert (
        non_annotated_dataset_path.name == pseudo_labeled_dataset_info["metainfo"]["version"]
    ), f"Please check consistency between non_annotated dataset: {non_annotated_dataset_path} and info file: {pseudo_labeled_info_path}"
    scene_ids: List[str] = [
        d.name for d in non_annotated_dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    pseudo_label_generator = PseudoLabelGenerator3D(
        non_annotated_dataset_path=non_annotated_dataset_path,
        scene_ids=scene_ids,
        t4dataset_config=t4dataset_config,
        logger=logger,
    )

    # store frame length of each scene for checking consistency
    frame_length: Dict[str, int] = {scene_id: 0 for scene_id in scene_ids}

    progress_bar = ProgressBar(len(pseudo_labeled_dataset_info["data_list"]))
    for frame_info in pseudo_labeled_dataset_info["data_list"]:
        # get scene_id and frame_num from lidar_path
        scene_id, frame_num = _get_scene_and_frame_num(lidar_path=frame_info["lidar_points"]["lidar_path"])

        # check consistency between non_annotated dataset and info file.
        frame_length[scene_id] += 1
        assert (scene_id in scene_ids) and (
            frame_num + 1 == frame_length[scene_id]
        ), f"Please check the directory structure of your t4dataset and consistency between non_annotated dataset: {non_annotated_dataset_path} and info file: {pseudo_labeled_info_path}."

        # get ego2global for this frame
        ego2global = np.array(frame_info["ego2global"])

        # add pseudo label
        for pred_instance_3d in frame_info["pred_instances_3d"]:
            bbox: T4Box3D = _transform_pred_instance_to_global_t4box(
                pred_instance_3d["bbox_3d"],
                pred_instance_3d["velocity"],
                pred_instance_3d["bbox_score_3d"],
                pseudo_labeled_dataset_info["metainfo"]["classes"][pred_instance_3d["bbox_label_3d"]],
                pred_instance_3d["instance_id_3d"],
                ego2global,
                frame_info["timestamp"],
            )

            pseudo_label_generator.add_3d_annotation_object(
                scene_id=scene_id,
                frame_num=frame_num,
                t4box3d=bbox,
            )

        progress_bar.update()

    # validate and save pseudo labeled t4dataset
    pseudo_label_generator.dump(overwrite=overwrite)
    logger.info("All process success.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pseudo labeled t4dataset from pseudo labeled info file")
    parser.add_argument(
        "t4dataset_config",
        type=str,
        help="Path to yaml config file about T4dataset e.g. ./config/x2.yaml",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        required=True,
        help='Path to non-annotated dataset directory. e.g, "./data/t4dataset/pseudo_xx1/"',
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to pseudo labeled pkl file. e.g. ./data/t4dataset/info/pseudo_infos_raw_bevfusion.pkl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If stored, this code can overwrite sample_annotation.json even if t4dataset in root-path already have the annotation information.",
    )
    parser.add_argument(
        "--log-level",
        help="Set log level",
        default="INFO",
        choices=list(logging._nameToLevel.keys()),
    )
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    return parser.parse_args()


def main():
    init_default_scope("mmdet3d")

    # Parse command line arguments
    args = parse_args()

    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Check if input file exists
    root_path = Path(args.root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"Root path to non annotated dataset not found: {args.root_path}")

    # load t4dataset config
    with open(args.t4dataset_config, "r") as file:
        t4dataset_config: Dict[str, Any] = yaml.safe_load(file)

    logger: logging.Logger = setup_logger(args, name="create_pseudo_t4dataset")

    create_pseudo_t4dataset(
        pseudo_labeled_info_path=input_path,
        non_annotated_dataset_path=root_path,
        t4dataset_config=t4dataset_config,
        overwrite=args.overwrite,
        logger=logger,
    )


if __name__ == "__main__":
    main()
