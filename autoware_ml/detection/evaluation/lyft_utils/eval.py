"""https://github.com/lyft/nuscenes-devkit/blob/master/lyft_dataset_sdk/eval/detection/mAP_evaluation.py"""
from __future__ import annotations

from collections import defaultdict
import math
from os import path as osp
from typing import Dict, List

from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D as _Box3D
from lyft_dataset_sdk.eval.detection.mAP_evaluation import get_ap, get_ious, group_by_key
import mmengine
from mmengine.logging import print_log
import numpy as np
import pandas as pd


class Box3D(_Box3D):
    def __init__(self, **kwargs):
        kwargs["name"] = kwargs["detection_name"]
        kwargs["score"] = kwargs.get("detection_score", -1)
        super().__init__(**kwargs)

        self.area = self.length * self.width

    def get_iou_bev(self, other: Box3D):
        area_intersection = self.ground_bbox_coords.intersection(other.ground_bbox_coords).area
        area_union = self.area + other.area - area_intersection

        iou = np.clip(area_intersection / area_union, 0, 1)

        return iou

    def get_center_distance(self, other: Box3D):
        distance_2d = math.sqrt(
            (self.center_x - other.center_x) ** 2 + (self.center_y - other.center_y) ** 2
        )
        return distance_2d


def get_ious_bev(gt_boxes: List[Box3D], predicted_box: Box3D):
    return [predicted_box.get_iou_bev(x) for x in gt_boxes]


def get_center_distances(gt_boxes: List[Box3D], predicted_box: Box3D):
    return [predicted_box.get_center_distance(x) for x in gt_boxes]


def wrap_in_box(input: Dict):
    result = {}
    for key, value in input.items():
        result[key] = [Box3D(**x) for x in value]

    return result


def lyft_eval(
    nusc_det_annos: Dict,
    nusc_gt_annos: Dict,
    class_names: List[str],
    output_dir: str,
    logger=None,
    metric: str = "iou_bev",
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
):
    """Evaluation API for Lyft dataset.
    Args:
        nusc_det_annos (dict):
        gts (dict):
        output_dir (str): Output directory for output json files.
    Returns:
        dict[str, float]: The evaluation results.
    """
    assert metric in ["iou_3d", "iou_bev", "center_distance_2d"]

    # evaluate by lyft metrics
    gts = sum(nusc_gt_annos.values(), [])
    predictions = sum(nusc_det_annos.values(), [])

    print_log(f"Calculating mAP_{metric}@{thresholds}...", logger=logger)

    average_precisions = 100 * get_classwise_aps(gts, predictions, class_names, thresholds, metric)
    mAPs = np.nanmean(average_precisions, axis=0)
    mAPs_cate = np.mean(average_precisions, axis=1)
    mAP = np.nanmean(mAPs)

    metrics = dict()
    table_dict = defaultdict(dict)
    result_one_line_header = ""
    result_one_line = ""
    for i_c, class_name in enumerate(class_names):
        table_dict[class_name]["mAP"] = mAPs_cate[i_c]
        metrics[f"mAP_{class_name}"] = mAPs_cate[i_c]
        result_one_line_header += f"mAP_{class_name},"
        result_one_line += f"{mAPs_cate[i_c]},"
        for i_t, threshold in enumerate(thresholds):
            table_dict[class_name][f"AP@{threshold}"] = average_precisions[i_c, i_t]
            metrics[f"AP_{metric}_{threshold}_{class_name}"] = average_precisions[i_c, i_t]
            result_one_line_header += f"AP_{metric}_{threshold}_{class_name},"
            result_one_line += f"{average_precisions[i_c, i_t]},"

    table_df = pd.DataFrame.from_dict(table_dict, orient="index")
    table_df.index.name = "class_name"
    print_log(
        "----------------Lyft results-----------------\n"
        + table_df.to_markdown(mode="str", floatfmt=".2f"),
        logger=logger,
    )

    print_log(result_one_line_header, logger=logger)
    print_log(result_one_line, logger=logger)

    metrics["mAP"] = mAP
    mmengine.dump(table_dict, osp.join(output_dir, "lyft_metrics.json"))

    return metrics


def get_classwise_aps(gt, predictions, class_names, iou_thresholds, metric: str):
    """Returns an array with an average precision per class.
    Note: Ground truth and predictions should have the following format.
    .. code-block::
    gt = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207
                         fbb039a550991a5149214f98cec136ac',
        'translation': [974.2811881299899, 1714.6815014457964,
                        -23.689857123368846],
        'size': [1.796, 4.488, 1.664],
        'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
        'detection_name': 'car'
    }]
    predictions = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207
                         fbb039a550991a5149214f98cec136ac',
        'translation': [971.8343488872263, 1713.6816097857359,
                        -25.82534357061308],
        'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
        'rotation': [0.10913582721095375, 0.04099572636992043,
                     0.01927712319721745, 1.029328402625659],
        'detection_name': 'car',
        'detection_score': 0.3077029437237213
    }]
    Args:
        gt (list[dict]): list of dictionaries in the format described below.
        predictions (list[dict]): list of dictionaries in the format
            described below.
        class_names (list[str]): list of the class names.
        iou_thresholds (list[float]): IOU thresholds used to calculate
            TP / FN
    Returns:
        np.ndarray: an array with an average precision per class.
    """
    assert all([0 <= iou_th <= 1 for iou_th in iou_thresholds])

    gt_by_class_name = group_by_key(gt, "detection_name")
    pred_by_class_name = group_by_key(predictions, "detection_name")

    average_precisions = np.zeros((len(class_names), len(iou_thresholds)))

    for class_id, class_name in enumerate(class_names):
        if class_name in pred_by_class_name:
            recalls, precisions, average_precision = get_single_class_aps(
                gt_by_class_name[class_name],
                pred_by_class_name[class_name],
                iou_thresholds,
                metric,
            )
            average_precisions[class_id, :] = average_precision

    return average_precisions


def get_single_class_aps(gt, predictions, iou_thresholds, metric: str):
    """Compute recall and precision for all iou thresholds. Adapted from
    LyftDatasetDevkit.
    Args:
        gt (list[dict]): list of dictionaries in the format described above.
        predictions (list[dict]): list of dictionaries in the format \
            described below.
        iou_thresholds (list[float]): IOU thresholds used to calculate \
            TP / FN
    Returns:
        tuple[np.ndarray]: Returns (recalls, precisions, average precisions)
            for each class.
    """
    num_gts = len(gt)
    if num_gts == 0:
        recalls = aps = np.array([np.nan for _ in range(len(iou_thresholds))])
        precisions = np.zeros(len(iou_thresholds))
        return recalls, precisions, aps

    image_gts = group_by_key(gt, "sample_token")
    image_gts = wrap_in_box(image_gts)

    sample_gt_checked = {
        sample_token: np.zeros((len(boxes), len(iou_thresholds)))
        for sample_token, boxes in image_gts.items()
    }

    predictions = sorted(predictions, key=lambda x: x["detection_score"], reverse=True)

    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tps = np.zeros((num_predictions, len(iou_thresholds)))
    fps = np.zeros((num_predictions, len(iou_thresholds)))

    for prediction_index, prediction in enumerate(predictions):
        predicted_box = Box3D(**prediction)

        sample_token = prediction["sample_token"]

        if metric in ["iou_3d", "iou_bev"]:
            max_overlap = -np.inf
        elif metric in ["center_distance_3d", "center_distance_2d"]:
            max_overlap = np.inf
        else:
            raise ValueError(f"No metric: {metric}")
        jmax = -1

        if sample_token in image_gts:
            gt_boxes = image_gts[sample_token]
            # gt_boxes per sample
            gt_checked = sample_gt_checked[sample_token]
            # gt flags per sample
        else:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            if metric == "iou_3d":
                overlaps = get_ious(gt_boxes, predicted_box)
            elif metric == "iou_bev":
                overlaps = get_ious_bev(gt_boxes, predicted_box)
            elif metric == "center_distance_2d":
                overlaps = get_center_distances(gt_boxes, predicted_box)
            else:
                raise NotImplementedError(f"Invalid metric: {metric}")

            if metric in ["iou_3d", "iou_bev"]:
                max_overlap = np.max(overlaps)
                jmax = np.argmax(overlaps)
            elif metric in ["center_distance_3d", "center_distance_2d"]:
                max_overlap = np.min(overlaps)
                jmax = np.argmin(overlaps)
            else:
                raise ValueError(f"No metric: {metric}")

        for i, iou_threshold in enumerate(iou_thresholds):
            is_matched = False
            if metric in ["iou_3d", "iou_bev"]:
                is_matched = max_overlap > iou_threshold
            elif metric in ["center_distance_3d", "center_distance_2d"]:
                is_matched = max_overlap < iou_threshold
            else:
                raise ValueError(f"No metric: {metric}")

            if is_matched:
                if gt_checked[jmax, i] == 0:
                    tps[prediction_index, i] = 1.0
                    gt_checked[jmax, i] = 1
                else:
                    fps[prediction_index, i] = 1.0
            else:
                fps[prediction_index, i] = 1.0

    # compute precision recall
    fps = np.cumsum(fps, axis=0)
    tps = np.cumsum(tps, axis=0)

    recalls = tps / float(num_gts)
    # avoid divide by zero in case the first detection
    # matches a difficult ground truth
    precisions = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)

    aps = []
    for i in range(len(iou_thresholds)):
        recall = recalls[:, i]
        precision = precisions[:, i]
        assert np.all(0 <= recall) & np.all(recall <= 1)
        assert np.all(0 <= precision) & np.all(precision <= 1)

        rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
        precision = np.interp(rec_interp, recall, precision, right=0)
        recall = rec_interp

        ap = get_ap(recall, precision)
        aps.append(ap)

    aps = np.array(aps)

    return recalls, precisions, aps
