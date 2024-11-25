import argparse
import os
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from matplotlib import colormaps
from mmdet3d.registry import MODELS
from mmengine.config import Config
from mmengine.device import get_device
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, autocast, load_checkpoint
from torch import Tensor


def parse_args() -> argparse.Namespace:
    """
    Parse args.

    Returns:
        argparse.Namespace: argsparse objects
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        metavar="FILE",
        help="The config of model file or dataset loader.",
    )
    parser.add_argument(
        "product_config",
        metavar="FILE",
        help="The config for visualization.",
    )
    parser.add_argument(
        "--objects",
        type=str,
        default="prediction",
        choices=["prediction", "ground_truth"],
        help=
        "What objects you want to visualize. You choice from prediction, ground_truth.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=
        "If you choose prediction visualization, you need checkpoint file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Choose dataset from train, val and test.",
    )
    parser.add_argument(
        "--bbox-score",
        type=float,
        default=0.4,
        help="Score threshold if you choose prediction visualization.",
    )
    parser.add_argument(
        "--image-num",
        type=int,
        default=6,
        help="The number of images. 6 is default number for NuScenes dataset.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="The number of skip frames.",
    )
    parser.add_argument(
        "--fix-rotation",
        action="store_true",
        help="The option for fixing rotation bug. it needs for nuScenes data.",
    )
    rr.script_add_args(parser)

    args = parser.parse_args()
    return args


def update_config(
    cfg: dict[Any],
    cfg_product: dict[Any],
    args: argparse.Namespace,
) -> list[dict[Any]]:
    """
    Update config

    Args:
        cfg (dict[Any]): Model config dict of mmdetection3d
        cfg_product (dict[Any]): Product config dict
        args (argparse.Namespace): The argparse object

    Returns:
        list[dict[Any]]: Updated configs of cfg and cfg_product
    """

    # update batchsize
    cfg.val_dataloader.batch_size = 1
    cfg.test_dataloader.batch_size = 1

    # update multi sweep
    cfg = delete_multi_sweep(cfg)

    # update modality
    modality = dict(use_lidar=True, use_camera=True)
    cfg.train_dataloader.dataset.dataset.modality = modality
    cfg.val_dataloader.dataset.modality = modality
    cfg.test_dataloader.dataset.modality = modality

    # update camera_orders from args
    if args.image_num < len(cfg_product.camera_panels):
        cfg_product.camera_panels = cfg_product.camera_panels[0:args.image_num]

    return cfg, cfg_product


def get_class_color_list(
    class_colors: dict[Any],
    class_names: list[str],
) -> list[list[int]]:
    """
    Get correspondence between colors and objects.

    Args:
        class_colors (dict[Any]): Color setting
        class_names (list[str]): Use of class name

    Returns:
        list[list[int]]: Correspondence between colors and objects. The type is list[(R, G, B), ...]
    """
    class_color_list = []
    for class_name in class_names:
        class_color_list.append(class_colors[class_name])
    return class_color_list


def delete_multi_sweep(cfg: dict[Any]) -> dict[Any]:
    """
    Delete multi sweep setting.
    Args:
        cfg (dict[Any]): cfg object of mmdetection3d.

    Returns:
        dict[Any]: Updated cfg object.
    """
    fixed_pipeline = []
    for component in cfg.train_dataloader.dataset.dataset.pipeline:
        if component["type"] == "LoadPointsFromMultiSweeps":
            component["sweeps_num"] = 1
        fixed_pipeline.append(component)
    cfg.train_dataloader.dataset.pipeline = fixed_pipeline

    fixed_pipeline = []
    for component in cfg.val_dataloader.dataset.pipeline:
        if component["type"] == "LoadPointsFromMultiSweeps":
            component["sweeps_num"] = 1
        fixed_pipeline.append(component)
    cfg.val_dataloader.dataset.pipeline = fixed_pipeline

    fixed_pipeline = []
    for component in cfg.test_dataloader.dataset.pipeline:
        if component["type"] == "LoadPointsFromMultiSweeps":
            component["sweeps_num"] = 1
        fixed_pipeline.append(component)
    cfg.test_dataloader.dataset.pipeline = fixed_pipeline
    return cfg


def init_rerun(args: argparse.Namespace, camera_panels: list[str]):
    """
    Initialize rerun.io.

    Args:
        args (argparse.Namespace): Args object.
        camera_panels (list[str]): The name of camera panels
    """

    # set blueprint
    sensor_space_views = [
        rrb.Spatial2DView(
            name=sensor_name,
            origin=f"world/ego_vehicle/{sensor_name}",
        ) for sensor_name in camera_panels
    ]
    blueprint = rrb.Vertical(
        rrb.Spatial3DView(
            name="3D",
            origin="world",
            # Default for `ImagePlaneDistance` so that the pinhole frustum visualizations don't take up too much space.
            # defaults=[rr.components.ImagePlaneDistance(4.0)],
            # overrides={"world/ego_vehicle": [rr.components.AxisLength(5.0)]},
        ),
        rrb.Grid(*sensor_space_views),
        row_shares=[5, 2],
    )
    rr.script_setup(args, "visualization", default_blueprint=blueprint)

    # setup visualization
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def euler_to_quaternion(
    yaw: float,
    pitch: float,
    roll: float,
    fix_rotation: bool,
) -> list[float]:
    """
    Convert euler to quaternion.

    Args:
        yaw (float): Yaw angle.
        pitch (float): Pitch angle.
        roll (float): Roll angle.
        fix_rotation (bool): If true, only using yaw angle.

    Returns:
        list[float]: Quaternion
    """
    if fix_rotation:
        pitch = 0.0
        roll = 0.0
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def convert_gt_objects(
    data: dict[Any],
    frame_number: int,
) -> list[Tensor]:
    """
    Convert from dataset to list of ground truth object.

    Args:
        data (dict[Any]): Data of dataloader of mmdetection3d.
        frame_number (int): Frame number

    Returns:
        list[Tensor]: List of ground truth object.
    """
    if "gt_bboxes_labels" in data["data_samples"][0].eval_ann_info:
        bboxes = data["data_samples"][0].eval_ann_info["gt_bboxes_3d"]
        labels = data["data_samples"][0].eval_ann_info["gt_bboxes_labels"]
        return bboxes, labels
    else:
        print(f"frame {frame_number}: there is no objects")
        return None, None


def convert_pred_objects(
    outputs: dict[Any],
    bbox_score_threshold: float,
) -> list[Tensor]:
    """
    Convert from inference output to list of inference object.

    Args:
        outputs (dict[Any]): Inference outputs.
        bbox_score_threshold (float): The threshold of score

    Returns:
        list[Tensor]: List of 3d bounding box of inference results.
    """
    bboxes = outputs[0].pred_instances_3d["bboxes_3d"].tensor.detach().cpu()
    scores = outputs[0].pred_instances_3d["scores_3d"].detach().cpu()
    labels = outputs[0].pred_instances_3d["labels_3d"].detach().cpu()
    if bbox_score_threshold is not None:
        indices = scores >= bbox_score_threshold
        bboxes = bboxes[indices]
        scores = scores[indices]
        labels = labels[indices]
    return bboxes, labels, scores


def visualize_objects(
    bboxes: Tensor,
    labels: Tensor,
    scores: Tensor,
    fix_rotation: bool,
    object_colors: list[list[int]],
):
    """
    Visualize 3D objects

    Args:
        bboxes (Tensor): The tensor of bounding boxes.
        labels (Tensor): The tensor of labels.
        scores (Tensor): The tensor of scores.
        fix_rotation (bool): If true, only using yaw angle.
        object_colors (list[list[int]]): The color setting.
    """
    centers = []
    sizes = []
    quaternions = []
    colors = []
    rerun_label_texts = []
    for bbox in bboxes:
        bbox = bbox.to('cpu').detach().numpy().copy()
        size = bbox[3:6]
        sizes.append(size)
        center = bbox[0:3]
        # fixed center point at z-axis
        center[2] += size[2] / 2.0
        centers.append(center)
        rotation = euler_to_quaternion(*bbox[6:9], fix_rotation)
        quaternions.append(rr.Quaternion(xyzw=rotation))

    for label in labels:
        colors.append(object_colors[label])
    
    for score in scores:
        rerun_label_texts.append("Score: {0:.3f}".format(score))

    rr.log(
        "world/ego_vehicle/bbox",
        rr.Boxes3D(
            centers=centers,
            sizes=sizes,
            rotations=quaternions,
            class_ids=labels,
            colors=colors,
            labels=rerun_label_texts,
        ),
    )


def visualize_lidar(data: dict[Any], sensor_name: str):
    """
    Visualize LiDAR data.

    Args:
        data (dict[Any]): The data of dataloader.
        sensor_name (str): Sensor name of LiDAR
    """
    lidar = data["inputs"]["points"][0]
    # shape after transposing: (num_points, 3)
    points = lidar[:, :3]

    # color points based on height
    heights = points[:, 2].numpy()
    min_height = np.min(heights)
    max_height = np.max(heights)
    normalized_heights = (heights - min_height) / (max_height - min_height)
    colormap = colormaps['viridis']
    colors = colormap(normalized_heights)
    colors_rgb = (colors[:, :3] * 255).astype(np.int32)

    # visualize
    rr.log(f"world/ego_vehicle/{sensor_name}",
           rr.Points3D(points, colors=colors_rgb))


def visualize_camera(data: dict[Any], data_root: str):
    """
    Visualize camera data.

    Args:
        data (dict[Any]): The data of dataloader.
        data_root (str): The root data for camera data.
    """
    for img_path in data["data_samples"][0].img_path:
        if img_path is not None:
            img_path_list = img_path.split('/')
            panel_name = os.path.join(*img_path_list[3:5])

            full_path = os.path.join(data_root, img_path)
            rr.log(f"world/ego_vehicle/{panel_name}",
                   rr.ImageEncoded(path=full_path))


def main():
    args = parse_args()
    init_default_scope('mmdet3d')

    # create config
    cfg = Config.fromfile(args.config)
    cfg_product = Config.fromfile(args.product_config)

    cfg, cfg_product = update_config(cfg, cfg_product, args)
    class_color_list = get_class_color_list(cfg_product.class_colors,
                                            cfg.class_names)

    # build dataset
    dataset = Runner.build_dataloader(cfg.test_dataloader)

    # build model
    model = None
    if args.objects == "prediction":
        # build model and load checkpoint
        model = MODELS.build(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        model.to(get_device())
        model.eval()

    # init rerun visualization
    init_rerun(args, cfg_product.camera_panels)

    for frame_number, data in enumerate(dataset):
        if frame_number % args.skip_frames != 0:
            continue

        # set frame number
        rr.set_time_seconds("frame_number", frame_number * 0.1)

        # ego vehicle set (0, 0, 0)
        rr.log(
            "world/ego_vehicle",
            rr.Transform3D(
                translation=[0, 0, 0],
                rotation=rr.Quaternion(xyzw=[0, 0, 0, 1]),
                from_parent=False,
            ),
        )

        # bounding box
        bboxes = None
        labels = None
        if args.objects == "ground_truth":
            bboxes, labels = convert_gt_objects(data, frame_number)
            scores = np.ones_like(labels)
        elif args.objects == "prediction":
            with autocast(enabled=True):
                outputs = model.test_step(data)
            bboxes, labels, scores = convert_pred_objects(
                outputs,
                args.bbox_score,
            )
        if bboxes is not None:
            visualize_objects(
                bboxes,
                labels,
                scores,
                args.fix_rotation,
                class_color_list,
            )

        # lidar
        visualize_lidar(data, cfg_product.data_prefix["pts"])

        # camera
        visualize_camera(data, cfg.data_root)

    rr.script_teardown(args)


if __name__ == '__main__':
    main()
