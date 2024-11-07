import os
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Tuple, Union

import mmengine
import numpy as np
import pyquaternion
from mmdet3d.evaluation.metrics import NuScenesMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmengine import load
from mmengine.logging import MMLogger, print_log
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.utils.data_classes import Box as NuScenesBox

from autoware_ml.detection3d.evaluation.t4metric.evaluation import (
    T4DetectionConfig, T4DetectionEvaluation)
from autoware_ml.detection3d.evaluation.t4metric.loading import (
    t4metric_load_gt, t4metric_load_prediction)

__all__ = ["T4Metric"]


# [TODO] This class will refactor. We will rewrite T4Metrics
# using [autoware_perception_evaluation](https://github.com/tier4/autoware_perception_evaluation).
@METRICS.register_module()
class T4Metric(NuScenesMetric):
    """T4 format evaluation metric.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
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

        self.class_names = class_names
        self.version = version

        if name_mapping is None:
            self.class_names = [
                self.name_mapping.get(name, name) for name in self.class_names
            ]
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
        self.data_infos = load(
            self.ann_file, backend_args=self.backend_args)["data_list"]
        scene_tokens, directory_names = self._get_scene_info(self.data_infos)
        self.loaded_scenes, self.scene_tokens, self.directory_names = self._load_scenes(
            self.data_root,
            scene_tokens,
            directory_names,
        )

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
        self.eval_detection_configs = T4DetectionConfig.deserialize(
            eval_config_dict)

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
            # ['database_v1_0', '3a13032b-6045-4db4-8632-9c52c3dd2fd9', '0', 'data', 'LIDAR_CONCAT', '98.pcd.bin']
            directory_list = info["lidar_points"]["lidar_path"].split("/")
            # 'database_v1_0/3a13032b-6045-4db4-8632-9c52c3dd2fd9/0'
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
                print(
                    f"Skipped non-existing {scene_directory_path} in T4Metric")
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
        result_dict, tmp_dir = self.format_results(results, self.class_names,
                                                   self.jsonfile_prefix)
        metric_dict = {}

        if self.format_only and self.jsonfile_prefix:
            logger.info(
                f"results are saved in {os.path.basename(self.jsonfile_prefix)}"
            )
            return metric_dict

        all_preds = EvalBoxes()
        all_gts = EvalBoxes()

        for scene in self.scene_tokens:
            ap_dict, preds, gts = self.t4_evaluate(
                scene, result_dict, classes=self.class_names)

            all_preds = concatenate_eval_boxes(all_preds, preds)
            all_gts = concatenate_eval_boxes(all_gts, gts)

        ap_dict = self.t4_evaluate_all_scenes(result_dict, all_preds, all_gts,
                                              self.class_names, logger)
        for result in ap_dict:
            metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def t4_evaluate_all_scenes(
        self,
        result_dict: dict,
        preds: EvalBoxes,
        gts: EvalBoxes,
        classes: List[str],
        logger: MMLogger,
    ) -> Dict[str, float]:
        all_detail = dict()
        for name in result_dict:
            result_path = result_dict[name]
            output_dir = os.path.join(*os.path.split(result_path)[:-1])

            evaluator = T4DetectionEvaluation(
                config=self.eval_detection_configs,
                result_path=result_path,
                scene="",
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
            detail = self._create_detail(metrics, classes)
            all_detail.update(detail)
        return all_detail

    def t4_evaluate(
        self,
        scene_token: str,
        result_dict: dict,
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
        all_preds = EvalBoxes()
        all_gts = EvalBoxes()
        for name in result_dict:
            ret_dict, preds, gts = self._evaluate_scene(
                scene_token, result_dict[name], classes=classes)
            all_preds = concatenate_eval_boxes(all_preds, preds)
            all_gts = concatenate_eval_boxes(all_gts, gts)
            metric_dict.update(ret_dict)
        return metric_dict, all_preds, all_gts

    def _evaluate_scene(
        self,
        scene_token: str,
        result_path: str,
        classes: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, float], EvalBoxes, EvalBoxes]:
        """Evaluation for a single model in nuScenes protocol.

        Args:
            scene_token (str): Scene token.
            result_path (str): Path of the result file.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            result_name (str): Result name in the metric prefix.
                Defaults to 'pred_instances_3d'.

        Returns:
            Dict[str, float]: Dictionary of evaluation details.
        """

        output_dir = os.path.join(*os.path.split(result_path)[:-1])
        nusc = self.loaded_scenes[scene_token]

        gt_boxes = t4metric_load_gt(
            nusc,
            self.eval_detection_configs,
            scene_token,
            post_mapping_dict=self.name_mapping,
            filter_attributions=self.filter_attributes)
        preds, _ = t4metric_load_prediction(
            nusc,
            self.eval_detection_configs,
            result_path,
            self.eval_detection_configs.max_boxes_per_sample,
            verbose=True,
        )

        evaluator = T4DetectionEvaluation(
            config=self.eval_detection_configs,
            result_path=result_path,
            scene=scene_token,
            output_dir=output_dir,
            verbose=False,
            ground_truth_boxes=gt_boxes,
            prediction_boxes=preds,
        )
        _, metrics_table = evaluator.run_and_save_eval()

        # record metrics
        metrics = load(os.path.join(output_dir, "metrics_summary.json"))
        detail = self._create_detail(metrics, classes)

        return detail, preds, gt_boxes

    def _create_detail(self, metrics: dict,
                       classes: Optional[List[str]]) -> Dict[str, float]:
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

        for name in results[0]:
            if "pred" in name and "3d" in name and name[0] != "_":
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                box_type_3d = type(results_[0]["bboxes_3d"])
                if box_type_3d == LiDARInstance3DBoxes:
                    result_dict[name] = self._format_lidar_bbox(
                        results_, sample_idx_list, classes, tmp_file_)
                elif box_type_3d == CameraInstance3DBoxes:
                    result_dict[name] = self._format_camera_bbox(
                        results_, sample_idx_list, classes, tmp_file_)

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
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_idx],
                                             boxes, classes,
                                             self.eval_detection_configs)
            for i, box in enumerate(boxes):
                name = classes[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
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
        print(f"Results writes to {res_path}")
        mmengine.dump(nusc_submissions, res_path)
        return res_path


def concatenate_eval_boxes(eval_boxes1: EvalBoxes,
                           eval_boxes2: EvalBoxes) -> EvalBoxes:
    """
    Concatenates two EvalBoxes instances into a new EvalBoxes instance.

    Parameters:
    - eval_boxes1: The first EvalBoxes instance.
    - eval_boxes2: The second EvalBoxes instance.

    Returns:
    - A new EvalBoxes instance containing boxes from both input instances.
    """
    new_eval_boxes = EvalBoxes(
    )  # Initialize a new instance to hold the combined boxes

    # Function to add boxes from an EvalBoxes instance to the new instance
    def add_from_instance(instance: EvalBoxes):
        for sample_token, boxes in instance.boxes.items():
            new_eval_boxes.add_boxes(sample_token,
                                     boxes)  # Add boxes for each sample token

    # Add boxes from both instances to the new instance
    add_from_instance(eval_boxes1)
    add_from_instance(eval_boxes2)

    return new_eval_boxes


def lidar_nusc_box_to_global(
        info: dict, boxes: List[NuScenesBox], classes: List[str],
        eval_configs: DetectionConfig) -> List[NuScenesBox]:
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
        box.rotate(
            pyquaternion.Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07))
        box.translate(lidar2ego[:3, 3])
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        ego2global = np.array(info["ego2global"])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
    return box_list


def output_to_nusc_box(
        detection: dict) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
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
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
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
            q1 = pyquaternion.Quaternion(
                axis=[0, 0, 1], radians=nus_box_yaw[i])
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
        raise NotImplementedError(
            f"Do not support convert {type(bbox3d)} bboxes "
            "to standard NuScenesBoxes.")

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
    header_str = ("| " + " | ".join(header[i].ljust(col_widths[i])
                                    for i in range(len(header))) + " |\n")
    # Format table_middle
    table_middle_str = "|" + " ---- |" * len(header) + "\n"

    # Format data rows
    rows = []
    for row in data:
        row_str = ("| " +
                   " | ".join("{{:<{}}}".format(col_widths[i]).format(row[i])
                              for i in range(len(row))) + " |\n")
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
