from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np
from mmdet.evaluation.functional import eval_map, eval_recalls
from mmdet.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log


@METRICS.register_module()
class TLRFineDetectorEvaluator(BaseMetric):

    def __init__(
        self,
        classes,
        bbox_width_thres=33,
        proposal_nums=(100, 300, 1000),
        iou_thrs=[0.5, 0.6, 0.7, 0.8, 0.9],
        metric="mAP",
        scale_ranges=None,
        logger=None,
        *args,
        **kwargs,
    ):
        self.classes = classes
        self.bbox_width_thres = bbox_width_thres
        self.scale_ranges = scale_ranges
        self.logger = logger
        self.metric = metric
        self.iou_thrs = iou_thrs
        self.proposal_nums = proposal_nums
        super().__init__(*args, **kwargs)

    def process(self, data_batch: dict, data_samples: Sequence[dict]):
        for data_sample in data_samples:
            result = dict()
            pred = data_sample["pred_instances"]
            result["img_id"] = data_sample["img_id"]
            result["bboxes"] = (
                (pred["bboxes"] * pred["bboxes"].new_tensor(data_sample["scale_factor"]).repeat(1, 2)).cpu().numpy()
            )
            result["scores"] = pred["scores"].cpu().numpy()
            result["labels"] = pred["labels"].cpu().numpy()
            result["arranged_bboxes"] = [result["bboxes"][result["labels"] == i] for i in range(len(self.classes))]

            # parse gt
            gt = dict()
            gt["width"] = data_sample["ori_shape"][1]
            gt["height"] = data_sample["ori_shape"][0]
            gt["img_id"] = data_sample["img_id"]

            gt_instance = data_sample["gt_instances"]
            gt["anns"] = gt_instance
            self.results.append((gt, result))

    def compute_metrics(self, results_info: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        ranges = [(0, np.inf)]
        if self.bbox_width_thres is not None:
            ranges.append((0, self.bbox_width_thres))
            ranges.append((self.bbox_width_thres, np.inf))
        for bbox_range in ranges:
            print_log(f"\n{'*' * 15}bbox range = {bbox_range}{'*' * 15}")
            if not isinstance(self.metric, str):
                assert len(self.metric) == 1
                self.metric = self.metric[0]
            allowed_metrics = ["mAP", "recall"]
            if self.metric not in allowed_metrics:
                raise KeyError(f"metric {self.metric} is not supported")
            annotations = []
            results = []
            for gt, preds in results_info:
                gt_bboxes = gt["anns"]["bboxes"]
                bbox_widths = gt_bboxes[:, 2] - gt_bboxes[:, 0]
                valid_indices = (bbox_widths >= bbox_range[0]) * (bbox_widths < bbox_range[1])
                annotations.append(
                    dict(
                        bboxes=gt["anns"]["bboxes"][valid_indices],
                        labels=gt["anns"]["labels"][valid_indices],
                    )
                )
                results.append(preds["arranged_bboxes"])
            eval_results = OrderedDict()
            self.iou_thrs = [self.iou_thrs] if isinstance(self.iou_thrs, float) else self.iou_thrs
            if self.metric == "mAP":
                assert isinstance(self.iou_thrs, list)
                mean_aps = []
                for iou_thr in self.iou_thrs:
                    print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                    mean_ap, _ = eval_map(
                        results,
                        annotations,
                        scale_ranges=self.scale_ranges,
                        iou_thr=iou_thr,
                        dataset=self.classes,
                        logger=self.logger,
                    )
                    mean_aps.append(mean_ap)
                    eval_results[f"AP{int(iou_thr * 100):02d}"] = round(mean_ap, 3)
                eval_results["mAP"] = sum(mean_aps) / len(mean_aps)
            elif self.metric == "recall":
                gt_bboxes = [ann["bboxes"] for ann in annotations]
                recalls = eval_recalls(
                    gt_bboxes,
                    results,
                    self.proposal_nums,
                    self.iou_thrs,
                    logger=self.logger,
                )
                for i, num in enumerate(self.proposal_nums):
                    for j, iou in enumerate(self.iou_thrs):
                        eval_results[f"recall@{num}@{iou}"] = recalls[i, j]
                if recalls.shape[1] > 1:
                    ar = recalls.mean(axis=1)
                    for i, num in enumerate(self.proposal_nums):
                        eval_results[f"AR@{num}"] = ar[i]
        return eval_results
