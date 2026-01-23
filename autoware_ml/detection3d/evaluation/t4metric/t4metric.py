import csv
import json
import os
import tempfile
from collections import defaultdict
from copy import deepcopy
from os import path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import pyquaternion
import torch
from mmdet3d.evaluation.metrics import NuScenesMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmengine import load
from mmengine.logging import MMLogger, print_log
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.utils.data_classes import Box as NuScenesBox

from autoware_ml.detection3d.evaluation.t4metric.evaluation import T4DetectionConfig, T4DetectionEvaluation
from autoware_ml.detection3d.evaluation.t4metric.loading import T4Box, add_center_dist, filter_eval_boxes

__all__ = ["T4Metric"]
_UNKNOWN = "unknown"


# [TODO] This class will refactor. We will rewrite T4Metrics
# using [autoware_perception_evaluation](https://github.com/tier4/autoware_perception_evaluation).
@METRICS.register_module()
class T4Metric(NuScenesMetric):
    """T4 format evaluation metric."""

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        checkpoint_path: Optional[Union[Path, str]] = None,
        save_csv: bool = False,
        dataset_name: str = "base",
        filter_attributes: Optional[List[Tuple[str, str]]] = None,
        metric: Union[str, List[str]] = "bbox",
        modality: dict = dict(use_camera=False, use_lidar=True),
        prefix: Optional[str] = None,
        format_only: bool = False,
        jsonfile_prefix: Optional[str] = None,
        eval_version: str = "detection_cvpr_2019",
        collect_device: str = "cpu",
        backend_args: Optional[dict] = None,
        class_names: List[str] = [],
        eval_class_range: Dict[str, int] = dict(),
        name_mapping: Optional[dict] = None,
        version: str = "",
    ) -> None:
        """
        Args:
            data_root (str):
                Path of dataset root.
            ann_file (str):
                Path of annotation file.
            save_csv (bool): Set True to save metrics to csv files.
            dataset_name (str):
                Dataset name running this T4Metric.
            filter_attributes (str)
                Filter out GTs with certain attributes. For example, [['vehicle.bicycle',
                'vehicle_state.parked']].
            metric (str or List[str]):
                Metrics to be evaluated. Defaults to 'bbox'.
            modality (dict):
                Modality to specify the sensor data used as input.
                Defaults to dict(use_camera=False, use_lidar=True).
            prefix (str, optional):
                The prefix that will be added in the metric
                names to disambiguate homonymous metrics of different evaluators.
                If prefix is not provided in the argument, self.default_prefix will
                be used instead. Defaults to None.
            format_only (bool):
                Format the output results without perform
                evaluation. It is useful when you want to format the result to a
                specific format and submit it to the test server.
                Defaults to False.
            jsonfile_prefix (str, optional):
                The prefix of json files including the
                file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Defaults to None.
            eval_version (str):
                Configuration version of evaluation.
                Defaults to 'detection_cvpr_2019'.
            collect_device (str):
                Device name used for collecting results from
                different ranks during distributed training. Must be 'cpu' or 'gpu'.
                Defaults to 'cpu'.
            backend_args (dict, optional):
                Arguments to instantiate the corresponding backend. Defaults to None.
            class_names (List[str], optional):
                The class names. Defaults to [].
            eval_class_range (Dict[str, int]):
                The range of each class
            name_mapping (dict, optional):
                The data class mapping, applied to ground truth during evaluation.
                Defaults to None.
            version (str, optional):
                The version of the dataset. Defaults to "".
        """

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metric=metric,
            modality=modality,
            prefix=prefix,
            format_only=format_only,
            jsonfile_prefix=jsonfile_prefix,
            eval_version=eval_version,
            collect_device=collect_device,
            backend_args=backend_args,
        )
        self.save_csv = save_csv
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.version = version
        self.checkpoint_path = checkpoint_path

        if name_mapping is None:
            self.class_names = [self.name_mapping.get(name, name) for name in self.class_names]
        else:
            self.name_mapping = name_mapping

        for name in self.class_names:
            if name not in eval_class_range:
                raise RuntimeError("missing range value")
        self.eval_class_range = eval_class_range
        self.filter_attributes = filter_attributes
        if self.filter_attributes is None:
            print_log("No attribute filtering is applied!")

        # load annotations
        self.data_infos = load(self.ann_file, backend_args=self.backend_args)["data_list"]
        scene_tokens, directory_names = self._get_scene_info(self.data_infos)
        self.loaded_scenes, self.scene_tokens, self.directory_names = self._load_scenes(
            self.data_root,
            scene_tokens,
            directory_names,
        )

        # Sample token to scene tokens mapping
        self.sample_token_to_scene_tokens = {}
        for scene_token in self.scene_tokens:
            nusc = self.loaded_scenes[scene_token]
            # Get all sample tokens in the DB.
            for sample in nusc.sample:
                self.sample_token_to_scene_tokens[sample["token"]] = scene_token

        # Load eval detection config
        eval_config_dict = {
            "class_names": tuple(self.class_names),
            "class_range": self.eval_class_range,
            "dist_fcn": "center_distance",
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "max_boxes_per_sample": 500,
            "mean_ap_weight": 5,
        }
        self.eval_detection_configs = T4DetectionConfig.deserialize(eval_config_dict)

        self.pred_instances_3d_key = "pred_instances_3d"
        self.gt_instances_3d_key = "gt_instances_3d"

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample["pred_instances_3d"]
            pred_2d = data_sample["pred_instances"]
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to("cpu")
            result[self.pred_instances_3d_key] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to("cpu")
            result["pred_instances"] = pred_2d
            sample_idx = data_sample["sample_idx"]
            result["sample_idx"] = sample_idx

            result["current_time"] = data_sample["timestamp"]
            result["scene_id"] = self._parse_scene_id(data_sample["lidar_path"])

            result[self.gt_instances_3d_key] = self._parse_ground_truth_from_sample(data_sample)
            self.results.append(result)

    def _parse_scene_id(self, lidar_path: str) -> str:
        """Parse scene ID from the LiDAR file path.

        Removes the `data_root` prefix and the trailing `/data` section.

        Args:
            lidar_path (str): The full file path of the LiDAR data.
            Example of the lidar_path: 'db_j6_v1/43e6e09a-93ce-488f-8f40-515187bc2753/2/data/LIDAR_CONCAT/0.pcd.bin'

        Returns:
            str: The extracted scene ID, or "unknown" if extraction fails.
            Example of the extracted scene ID: 'db_j6_v1/43e6e09a-93ce-488f-8f40-515187bc2753/2'
        """
        # TODO(vividf): This will be eventually moved to t4_devkit

        if not lidar_path or not lidar_path.startswith(self.data_root):
            return _UNKNOWN

        # Remove the data_root prefix
        relative_path = lidar_path[len(self.data_root) :].lstrip("/")  # Remove leading slash if exists
        path_parts = relative_path.split("/")

        # Extract scene ID before "data" section
        try:
            data_index = path_parts.index("data")
            return "/".join(path_parts[:data_index])
        except ValueError:
            return _UNKNOWN

    def _parse_ground_truth_from_sample(self, data_sample: Dict[str, Any]) -> dict:
        """Parses ground truth objects from the given data sample.

        Args:
            data_sample (Dict[str, Any]): A dictionary containing the ground truth data,
                                        including 3D bounding boxes, labels, and point counts.

        Returns:
            Ground truth bboxes with the following keys:
            {"bboxes_3d", "scores_3d", "labels_3d", "num_lidar_pts"}.
        """
        # Extract evaluation annotation info for the current sample
        eval_info: dict = data_sample.get("eval_ann_info", {})

        # gt_bboxes_3d: LiDARInstance3DBoxes with tensor of shape (N, 9)
        # Format per box: [x, y, z, l, w, h, yaw, vx, vy]
        gt_bboxes_3d: LiDARInstance3DBoxes = eval_info.get("gt_bboxes_3d", LiDARInstance3DBoxes([]))
        # bboxes: np.ndarray = gt_bboxes_3d.tensor.cpu().numpy()

        # gt_labels_3d: (N,) array of class indices (e.g., [0, 1, 2, 3, ...])
        gt_labels_3d: np.ndarray = eval_info.get("gt_labels_3d", np.array([]))

        # num_lidar_pts: (N,) array of int, number of LiDAR points inside each GT box
        num_lidar_pts: np.ndarray = eval_info.get("num_lidar_pts", np.array([]))

        return {
            "bboxes_3d": gt_bboxes_3d,
            "scores_3d": np.ones(len(gt_bboxes_3d), dtype=np.float32),
            "labels_3d": gt_labels_3d,
            "num_lidar_pts": num_lidar_pts,
        }

    @staticmethod
    def _get_scene_info(data_infos: List[dict]) -> Tuple[List[str], List[str]]:
        """Get scene tokens and directory names from data infos.

        Args:
            data_infos (List[dict]): The data infos.

        Returns:
            List[str]: The scene tokens.
            List[str]: The directory names.
        """
        scene_tokens = []
        directories = []
        for info in data_infos:
            scene_token = info["scene_token"]
            # ['db_jpntaxi_v1', '3a13032b-6045-4db4-8632-9c52c3dd2fd9', '0', 'data', 'LIDAR_CONCAT', '98.pcd.bin']
            directory_list = info["lidar_points"]["lidar_path"].split("/")
            # 'db_jpntaxi_v1/3a13032b-6045-4db4-8632-9c52c3dd2fd9/0'
            directory = osp.join(*directory_list[0:3])
            if directory not in directories:
                scene_tokens.append(scene_token)
                directories.append(directory)
        return scene_tokens, directories

    @staticmethod
    def _load_scenes(
        data_root: str,
        scene_tokens: List[str],
        directory_names: List[str],
    ) -> Tuple[Dict[str, NuScenes], List[str], List[str]]:
        """Load scenes from data infos.

        Returns:
            Dict[str, NuScenes]: The loaded scenes.
        """

        loaded_dirs = {}
        existing_dirs = []
        existing_scenes = []
        for directory, scene_token in zip(directory_names, scene_tokens):
            scene_directory_path = os.path.join(data_root, directory)
            if os.path.exists(scene_directory_path):
                loaded_dirs[scene_token] = NuScenes(
                    version="annotation",
                    dataroot=scene_directory_path,
                    verbose=False,
                )
                existing_dirs.append(directory)
                existing_scenes.append(scene_token)
            else:
                print(f"Skipped non-existing {scene_directory_path} in T4Metric")
        return loaded_dirs, existing_scenes, existing_dirs

    def compute_metrics(
        self,
        results: List[dict],
    ) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        result_dict, tmp_dir = self.format_results(results, self.class_names, self.jsonfile_prefix)
        metric_dict = {}

        if self.format_only and self.jsonfile_prefix:
            logger.info(f"results are saved in {os.path.basename(self.jsonfile_prefix)}")
            return metric_dict

        ap_dict, scene_preds, scene_gts = self._evaluate_scenes(
            result_path=result_dict[self.pred_instances_3d_key],
            gt_result_path=result_dict[self.gt_instances_3d_key],
            classes=self.class_names,
        )

        # Flatten preds and gts from scene level {scene_token: EvalBoxes} to global level EvalBoxes
        all_preds = flatten_scene_eval_boxes(scene_eval_boxes=scene_preds)

        all_gts = flatten_scene_eval_boxes(scene_eval_boxes=scene_gts)

        logger.info("==== GT boxes info after filtering in evaluation ====")
        class_names = {}
        eval_boxes = all_gts.boxes
        for boxes in eval_boxes.values():
            for box in boxes:
                if box.detection_name not in class_names:
                    class_names[box.detection_name] = 1
                else:
                    class_names[box.detection_name] += 1

        for class_name, value in class_names.items():
            logger.info(f"class: {class_name}: {value}")
        logger.info("===== End of GT Boxes info ====")

        logger.info("==== Est boxes info after filtering in evaluation ====")
        class_names = {}
        eval_boxes = all_preds.boxes
        for boxes in eval_boxes.values():
            for box in boxes:
                if box.detection_name not in class_names:
                    class_names[box.detection_name] = 1
                else:
                    class_names[box.detection_name] += 1

        for class_name, value in class_names.items():
            logger.info(f"class: {class_name}: {value}")
        logger.info("===== End of Est Boxes info ====")

        ap_dict = self.t4_evaluate_all_scenes(
            result_dict[self.pred_instances_3d_key], all_preds, all_gts, self.class_names, logger
        )
        for result in ap_dict:
            metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def t4_evaluate_all_scenes(
        self,
        result_path: str,
        preds: EvalBoxes,
        gts: EvalBoxes,
        classes: List[str],
        logger: MMLogger,
    ) -> Dict[str, float]:

        all_detail = dict()
        output_dir = os.path.join(*os.path.split(result_path)[:-1])

        evaluator = T4DetectionEvaluation(
            config=self.eval_detection_configs,
            result_path=result_path,
            scene="all",
            output_dir=output_dir,
            verbose=False,
            ground_truth_boxes=gts,
            prediction_boxes=preds,
        )
        _, metrics_table = evaluator.run_and_save_eval()

        print_metrics_table(
            metrics_table["header"],
            metrics_table["data"],
            metrics_table["total_mAP"],
            type(self).__name__,
            len(self.loaded_scenes),
            logger,
        )
        # record metrics
        metrics = load(os.path.join(output_dir, "metrics_summary.json"))
        detail = self._create_detail(metrics=metrics, classes=classes, logger=logger, save_csv=self.save_csv)
        all_detail.update(detail)
        return all_detail

    def _evaluate_scenes(
        self,
        result_path: dict,
        gt_result_path: dict,
        classes: List[str],
    ) -> Tuple[Dict[str, float], EvalBoxes, EvalBoxes]:
        """Evaluate the results in T4 format.

        Args:
            scene_token (str): The scene token.
            result_dict (dict): Formatted results of the dataset.
            classes (List[str]): The class names.
            logger (MMLogger): The logger.

        Returns:
            Dict[str, float]: The evaluation results.
        """
        metric_dict = dict()
        scene_preds, _ = self._load_eval_boxes(
            result_path=result_path,
            max_boxes_per_sample=self.eval_detection_configs.max_boxes_per_sample,
            verbose=True,
        )

        scene_gts, _ = self._load_eval_boxes(result_path=gt_result_path, max_boxes_per_sample=0, verbose=True)

        # TODO(KokSeang): Add Scene-level metrics

        return metric_dict, scene_preds, scene_gts

    def _load_eval_boxes(
        self, result_path: str, max_boxes_per_sample: int, verbose: bool = True
    ) -> Tuple[Dict[str, EvalBoxes], Dict]:
        """
        Modified version of https://github.com/nutonomy/nuscenes-devkit/blob/9b165b1018a64623b65c17b64f3c9dd746040f36/python-sdk/nuscenes/eval/common/loaders.py#L21
        adds name mapping capabilities
        """
        # Load from file and check that the format is correct.
        with open(result_path) as f:
            data = json.load(f)

        assert "results" in data, (
            "Error: No field `results` in result file. Please note that the result format changed."
            "See https://www.nuscenes.org/object-detection for more information."
        )

        # Deserialize results and get meta data.
        all_results = EvalBoxes.deserialize(data["results"], T4Box)
        all_objects = sum([len(boxes) for _, boxes in all_results.boxes.items()])

        meta = data["meta"]
        if verbose:
            print_log(
                "Loaded results from {}. Found detections for {} samples, total_boxes: {}.".format(
                    result_path, len(all_results.sample_tokens), all_objects
                )
            )

        # Check that each sample has no more than x predicted boxes.
        if max_boxes_per_sample > 0:
            for sample_token in all_results.sample_tokens:
                assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, (
                    "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample
                )

        # Flatten to a dict of {scene_token: EvalBoxes}
        scene_eval_boxes = defaultdict(EvalBoxes)
        for sample_token in all_results.sample_tokens:
            scene_token = self.sample_token_to_scene_tokens[sample_token]
            scene_eval_boxes[scene_token].add_boxes(sample_token=sample_token, boxes=all_results.boxes[sample_token])

        for scene_token, eval_boxes in scene_eval_boxes.items():
            nusc = self.loaded_scenes[scene_token]
            eval_boxes = add_center_dist(nusc, eval_boxes)

            # Filter boxes (distance, points per box, etc.).
            eval_boxes = filter_eval_boxes(nusc, eval_boxes, self.eval_detection_configs.class_range, verbose=verbose)

        return scene_eval_boxes, meta

    def _write_to_csv(self, header, data, csv_filename: str, logger) -> None:
        """"""
        with open(csv_filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in data:
                writer.writerow(row)
        print_log(f"Saved in {csv_filename}", logger=logger)

    def _create_detail(
        self, metrics: dict, classes: Optional[List[str]], logger=None, save_csv: bool = False
    ) -> Dict[str, float]:
        """Create a dictionary to store the details of the evaluation.

        Returns:
            Dict[str, float]: The dictionary to store the details of the evaluation.
        """
        detail = dict()
        metric_prefix = "T4Metric"
        if classes is not None:
            for name in classes:
                for k, v in metrics["label_aps"][name].items():
                    val = float(f"{v:.4f}")
                    detail[f"{metric_prefix}/{name}_AP_dist_{k}"] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float(f"{v:.4f}")
                    detail[f"{metric_prefix}/{name}_{k}"] = val
                for k, v in metrics["tp_errors"].items():
                    val = float(f"{v:.4f}")
                    detail[f"{metric_prefix}/{self.ErrNameMapping[k]}"] = val

        detail[f"{metric_prefix}/NDS"] = metrics["nd_score"]
        detail[f"{metric_prefix}/mAP"] = metrics["mean_ap"]

        if save_csv:
            log_file_path = logger.handlers[1].baseFilename
            log_dir = os.path.dirname(log_file_path)

            score_headers = ["NDS", "mAP"]
            score_data = [float(f"{metrics['nd_score']:.4f}"), float(f"{metrics['mean_ap']:.4f}")]
            for k, v in metrics["tp_errors"].items():
                val = float(f"{v:.4f}")
                score_headers.append(self.ErrNameMapping[k])
                score_data.append(val)

            csv_filename = osp.join(log_dir, f"scores_{self.dataset_name}.csv")
            self._write_to_csv(header=score_headers, data=[score_data], csv_filename=csv_filename, logger=logger)

            class_ap_headers = ["class", "mAP"]
            mean_ap = metrics["mean_dist_aps"]
            class_ap_data = []
            if classes is not None:
                first_class = classes[0]
                for k in metrics["label_aps"][first_class].keys():
                    header = f"AP_dist_{k}"
                    class_ap_headers.append(header)

                for k in metrics["label_tp_errors"][first_class].keys():
                    class_ap_headers.append(k)

                for name in classes:
                    ap_data = [name, float(f"{mean_ap[name] * 100.0:.4f}")]

                    for k, v in metrics["label_aps"][name].items():
                        val = float(f"{v*100:.4f}")
                        ap_data.append(val)

                    for k, v in metrics["label_tp_errors"][name].items():
                        val = float(f"{v:.4f}")
                        ap_data.append(val)

                    class_ap_data.append(ap_data)

                csv_filename = osp.join(log_dir, f"class_ap_{self.dataset_name}.csv")
                self._write_to_csv(
                    header=class_ap_headers, data=class_ap_data, csv_filename=csv_filename, logger=logger
                )

        return detail

    def format_results(
        self,
        results: List[dict],
        classes: Optional[List[str]] = None,
        jsonfile_prefix: Optional[str] = None,
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the mmdet3d results to standard NuScenes json file.

        Args:
            results (List[dict]): Testing results of the dataset.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of json files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.

        Returns:
            tuple: Returns (result_dict, tmp_dir), where ``result_dict`` is a
            dict containing the json filepaths, ``tmp_dir`` is the temporal
            directory created for saving json files when ``jsonfile_prefix`` is
            not specified.
        """
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_dict = dict()
        sample_idx_list = [result["sample_idx"] for result in results]

        # pred_instances_3d
        print(f"\nFormating bboxes of {self.pred_instances_3d_key}")
        results_ = [out[self.pred_instances_3d_key] for out in results]
        tmp_file_ = osp.join(jsonfile_prefix, self.pred_instances_3d_key)
        box_type_3d = type(results_[0]["bboxes_3d"])
        if box_type_3d == LiDARInstance3DBoxes:
            result_dict[self.pred_instances_3d_key] = self._format_lidar_bbox(
                results_, sample_idx_list, classes, tmp_file_
            )
        elif box_type_3d == CameraInstance3DBoxes:
            result_dict[self.pred_instances_3d_key] = self._format_camera_bbox(
                results_, sample_idx_list, classes, tmp_file_
            )

        # gt
        print(f"\nFormating gt bboxes of {self.gt_instances_3d_key}")
        results_ = [out[self.gt_instances_3d_key] for out in results]
        tmp_file_ = osp.join(jsonfile_prefix, self.gt_instances_3d_key)
        box_type_3d = type(results_[0]["bboxes_3d"])
        if box_type_3d == LiDARInstance3DBoxes:
            result_dict[self.gt_instances_3d_key] = self._format_gt_lidar_bbox(
                results_, sample_idx_list, classes, tmp_file_
            )
        else:
            raise NotImplementedError

        return result_dict, tmp_dir

    def _format_lidar_bbox(
        self,
        results: List[dict],
        sample_idx_list: List[int],
        classes: Optional[List[str]] = None,
        jsonfile_prefix: Optional[str] = None,
    ) -> str:
        """Convert the results to the standard format.

        Args:
            results (List[dict]): Testing results of the dataset.
            sample_idx_list (List[int]): List of result sample idx.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by modifying the
                jsonfile_prefix. Defaults to None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print("Start to convert detection format...")
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes, attrs = output_to_nusc_box(det)
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos[sample_idx]["token"]
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_idx], boxes, classes, self.eval_detection_configs)
            for i, box in enumerate(boxes):
                name = classes[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = self.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }
        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print_log(f"Results writes to {res_path}")
        mmengine.dump(nusc_submissions, res_path)
        return res_path

    def _format_gt_lidar_bbox(
        self,
        results: List[dict],
        sample_idx_list: List[int],
        classes: Optional[List[str]] = None,
        jsonfile_prefix: Optional[str] = None,
    ) -> str:
        """Convert the gt detections to the standard format.

        Args:
            results (List[dict]): Testing results of the dataset.
            sample_idx_list (List[int]): List of result sample idx.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by modifying the
                jsonfile_prefix. Defaults to None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print("Start to convert GT detection format...")
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes, attrs = output_to_nusc_box(det)
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos[sample_idx]["token"]
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_idx], boxes, classes, self.eval_detection_configs)
            for i, box in enumerate(boxes):
                name = classes[box.label]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attrs,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }
        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "gt_t4dataset.json")
        print_log(f"Results writes to {res_path}")
        mmengine.dump(nusc_submissions, res_path)
        return res_path


def flatten_scene_eval_boxes(scene_eval_boxes: Dict[str, EvalBoxes]) -> EvalBoxes:
    """Flatten per-scene EvalBoxes into a single EvalBoxes instance.
    Args:
        scene_eval_boxes (Dict[str, EvalBoxes]): Mapping from scene identifiers
            (e.g., scene tokens) to their corresponding EvalBoxes instances.
    Returns:
        EvalBoxes: A new EvalBoxes instance containing all boxes from every
        scene in ``scene_eval_boxes``, keyed by their sample tokens.
    """
    all_eval_boxes = EvalBoxes()
    for eval_boxes in scene_eval_boxes.values():
        for sample_token, boxes in eval_boxes.boxes.items():
            all_eval_boxes.add_boxes(sample_token, boxes)
    return all_eval_boxes


def concatenate_eval_boxes(
    eval_boxes1: EvalBoxes,
    eval_boxes2: EvalBoxes,
) -> EvalBoxes:
    """
    Concatenates two EvalBoxes instances into a new EvalBoxes instance.

    Parameters:
    - eval_boxes1: The first EvalBoxes instance.
    - eval_boxes2: The second EvalBoxes instance.

    Returns:
    - A new EvalBoxes instance containing boxes from both input instances.
    """
    new_eval_boxes = EvalBoxes()  # Initialize a new instance to hold the combined boxes

    # Function to add boxes from an EvalBoxes instance to the new instance
    def add_from_instance(instance: EvalBoxes):
        for sample_token, boxes in instance.boxes.items():
            new_eval_boxes.add_boxes(sample_token, boxes)  # Add boxes for each sample token

    # Add boxes from both instances to the new instance
    add_from_instance(eval_boxes1)
    add_from_instance(eval_boxes2)

    return new_eval_boxes


def lidar_nusc_box_to_global(
    info: dict,
    boxes: List[NuScenesBox],
    classes: List[str],
    eval_configs: DetectionConfig,
) -> List[NuScenesBox]:
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the calibration
            information.
        boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (List[str]): Mapped classes in the evaluation.
        eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.

    Returns:
        List[:obj:`DetectionConfig`]: List of standard NuScenesBoxes in the
        global coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        lidar2ego = np.array(info["lidar_points"]["lidar2ego"])
        box.rotate(pyquaternion.Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07))
        box.translate(lidar2ego[:3, 3])
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        ego2global = np.array(info["ego2global"])
        box.rotate(pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
    return box_list


def output_to_nusc_box(detection: dict) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - bboxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        Tuple[List[:obj:`NuScenesBox`], np.ndarray or None]: List of standard
        NuScenesBoxes and attribute labels.
    """
    bbox3d = detection["bboxes_3d"]
    scores = detection["scores_3d"]
    if isinstance(scores, torch.Tensor):
        scores = scores.numpy()

    labels = detection["labels_3d"]
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    attrs = None
    if "attr_labels" in detection:
        attrs = detection["attr_labels"].numpy()

    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    box_list = []

    if isinstance(bbox3d, LiDARInstance3DBoxes):
        # our LiDAR coordinate system -> nuScenes box coordinate system
        nus_box_dims = box_dims[:, [1, 0, 2]]
        for i in range(len(bbox3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*bbox3d.tensor[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (
            # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity,
            )
            box_list.append(box)
    elif isinstance(bbox3d, CameraInstance3DBoxes):
        # our Camera coordinate system -> nuScenes box coordinate system
        # convert the dim/rot to nuscbox convention
        nus_box_dims = box_dims[:, [2, 0, 1]]
        nus_box_yaw = -box_yaw
        for i in range(len(bbox3d)):
            q1 = pyquaternion.Quaternion(axis=[0, 0, 1], radians=nus_box_yaw[i])
            q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
            quat = q2 * q1
            velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity,
            )
            box_list.append(box)
    else:
        raise NotImplementedError(f"Do not support convert {type(bbox3d)} bboxes " "to standard NuScenesBoxes.")

    return box_list, attrs


def print_metrics_table(
    header: List[str],
    data: List[List[str]],
    total_mAP: str = "",
    metric_name: str = "",
    scene_length: int = 0,
    logger: Optional[MMLogger] = None,
) -> None:
    """
    Print a table of metrics.
    :param header: The header of the table.
    :param data: The data rows of the table.
    :param total_mAP: The total mAP to print at the end of the table.
    :param metric_name: The name of the metric to print at the top of the table.
    :param logger: The logger to use for printing. If None, print to stdout.
    """

    print_log(f"==== {scene_length} scenes ====", logger)

    # Combine header and data
    all_data = [header] + data

    # Calculate maximum width for each column
    col_widths: List[int] = []
    for i in range(len(header)):
        for row in all_data:
            if len(col_widths) <= i:
                col_widths.append(len(str(row[i])))
            else:
                col_widths[i] = max(col_widths[i], len(str(row[i])))

    # Format header
    header_str = "| " + " | ".join(header[i].ljust(col_widths[i]) for i in range(len(header))) + " |\n"
    # Format table_middle
    table_middle_str = "|" + " ---- |" * len(header) + "\n"

    # Format data rows
    rows = []
    for row in data:
        row_str = "| " + " | ".join("{{:<{}}}".format(col_widths[i]).format(row[i]) for i in range(len(row))) + " |\n"
        rows.append(row_str)

    # Print table
    print_str = f"\n------------- {metric_name} results -------------\n"
    print_str += header_str
    print_str += table_middle_str
    for line in rows:
        print_str += line
    if total_mAP != "":
        print_str += f"\nTotal mAP: {total_mAP}"
    print_log(print_str, logger)
