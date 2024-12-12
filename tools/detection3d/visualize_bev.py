import argparse
import os
from typing import List, Optional, Tuple

import mmengine
import numpy as np
from matplotlib import pyplot as plt
from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.config import Config
from mmengine.device import get_device
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, autocast, load_checkpoint

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=((xlim[1] - xlim[0]), (ylim[1] - ylim[0])))
    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmengine.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fname=fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-score", type=float, default=0.1)
    parser.add_argument("--out-dir", type=str, default="work_dirs/visualization")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_default_scope("mmdet3d")

    # create config
    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.test_dataloader.batch_size = 1

    # build dataset
    dataset = Runner.build_dataloader(cfg.test_dataloader)

    # build model and load checkpoint
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(get_device())
    model.eval()

    for i, data in enumerate(dataset):
        lidar_path = data["data_samples"][0].lidar_path.split("/")
        file_name = "_".join(lidar_path[3:8])

        with autocast(enabled=True):
            outputs = model.test_step(data)
        bboxes = outputs[0].pred_instances_3d["bboxes_3d"].tensor.detach().cpu()
        scores = outputs[0].pred_instances_3d["scores_3d"].detach().cpu()
        labels = outputs[0].pred_instances_3d["labels_3d"].detach().cpu()
        if args.bbox_score is not None:
            indices = scores >= args.bbox_score
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

        # lidar
        lidar = data["inputs"]["points"][0]
        fpath = os.path.join(args.out_dir, "lidar", f"{file_name}.png")
        visualize_lidar(
            fpath,
            lidar,
            bboxes=bboxes,
            labels=labels,
            xlim=[-120, 120],
            ylim=[-60, 60],
            classes=cfg.class_names,
        )


if __name__ == "__main__":
    main()
