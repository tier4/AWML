"""
modified from https://github.com/open-mmlab/OpenPCDet/blob/v0.5.2/pcdet/datasets/nuscenes/nuscenes_utils.py
"""
import operator
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from mmengine.logging import print_log
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from terminaltables import AsciiTable

class_mapping_kitti2nuscenes = {
    "Car": "car",
    "Cyclist": "bicycle",
    "Pedestrian": "pedestrian",
}

cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}


def boxes_lidar_to_nuscenes(det_info):
    boxes3d = det_info["boxes_lidar"]
    scores = det_info["score"]
    labels = det_info["pred_labels"]

    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, 3:6],  # wlh
            quat,
            label=labels[k],
            score=scores[k],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def transform_det_annos_to_nusc_annos(det_annos, velocity_threshold=0.2):
    """convert detections to nuscenes-format for evaluation
    modified from: https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/nuscenes/nuscenes_utils.py#L423

    Args:
        det_annos ([type]): [description]
        velocity_threshold (float, optional): [description]. Defaults to 0.2.

    det_annos = [
        {
            boxes_lidar: (np.ndarray, float),
            pred_labels: (np.ndarray, int),
            score: (np.ndarray, float),
            name: (np.ndarray, str),
            metadata: {
                token: (str)
            }
        },
        ...
    ]

    Returns:
        nusc_annos (dict[list]): [description]
    """

    nusc_annos = {}

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nuscenes(det)

        for k, box in enumerate(box_list):
            name = det["name"][k]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > velocity_threshold:
                if name in ["car", "construction_vehicle", "bus", "truck", "trailer"]:
                    attr = "vehicle.moving"
                elif name in ["bicycle", "motorcycle"]:
                    attr = "cycle.with_rider"
                else:
                    attr = None
            else:
                if name in ["pedestrian"]:
                    attr = "pedestrian.standing"
                elif name in ["bus"]:
                    attr = "vehicle.stopped"
                else:
                    attr = None
            attr = (
                attr
                if attr is not None
                else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            )
            nusc_anno = {
                "sample_token": det["metadata"]["token"],
                "translation": box.center.tolist(),
                "size": box.wlh.tolist(),
                "rotation": box.orientation.elements.tolist(),
                "velocity": box.velocity[:2].tolist(),
                "detection_name": name,
                "detection_score": box.score,
                "attribute_name": attr,
            }
            annos.append(nusc_anno)

        nusc_annos.update({det["metadata"]["token"]: annos})

    return nusc_annos


def format_nuscenes_metrics(metrics: Dict, class_names: List[str], version="default"):
    result = f"----------------nuScenes {version} results-----------------\n"

    result_dict: Dict[str, Dict[str, float]] = defaultdict(dict)
    for name in class_names:
        result_dict[name].update(
            {
                f"mAP": sum([v for v in metrics["label_aps"][name].values()])
                * 100
                / len(metrics["label_aps"][name])
            }
        )
        result_dict[name].update(
            {f"AP@{k}m": v * 100 for k, v in metrics["label_aps"][name].items()}
        )
        result_dict[name].update(
            {f"error@{k}": v for k, v in metrics["label_tp_errors"][name].items()}
        )
        result_dict[name].pop("error@attr_err")

    df = pd.DataFrame.from_dict(result_dict, orient="index")
    df.index.name = "class_name"
    result += df.to_markdown(mode="str", floatfmt=".2f")
    details = {}
    for key, val in metrics["tp_errors"].items():
        details[key] = val

    for key, val in metrics["mean_dist_aps"].items():
        details[f"mAP_{key}"] = val

    details.update(
        {
            "mAP": metrics["mean_ap"],
            "NDS": metrics["nd_score"],
        }
    )

    return result, details


def format_nuscenes_metrics_table(metrics, class_names, logger=None):
    name = class_names[0]
    thresholds = list(map(str, metrics["label_aps"][name].keys()))
    # not use error@attr
    errors = [x.split("_")[0] for x in list(metrics["label_tp_errors"][name].keys())][:4]

    APs_data = [
        ["class", "mAP"] + [f"AP@{e}" for e in thresholds] + [f"error@{e}" for e in errors]
    ]

    for name in class_names:
        ap_list = list(metrics["label_aps"][name].values())
        error_list = list(metrics["label_tp_errors"][name].values())[:4]

        mAP = round(metrics["mean_dist_aps"][name] * 100, 3)
        AP = (
            [name, mAP]
            + [round(ap * 100, 3) for ap in ap_list]
            + [round(e, 3) for e in error_list]
        )
        APs_data.append(AP)

    APs_table = AsciiTable(APs_data)
    # APs_table.inner_footing_row_border = True
    print_log("\n" + APs_table.table, logger=logger)

    details = {}
    for key, val in metrics["tp_errors"].items():
        details[key] = val

    details.update(
        {
            "mAP": metrics["mean_ap"],
            "NDS": metrics["nd_score"],
        }
    )

    return details
