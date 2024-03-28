import itertools
import json
import logging
import os
from os import path as osp
from typing import Any, Dict, List, Optional, Union
import warnings

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.evaluation.metrics.nuscenes_metric import output_to_nusc_box
from mmdet3d.structures import LiDARInstance3DBoxes
import mmengine
from mmengine.logging import print_log
import numpy as np
import pandas as pd
import pyquaternion
import torch

from autoware_ml.detection.datasets.class_mapping import map_dataset_classes
from autoware_ml.detection.evaluation import hdl_utils, lyft_utils, nuscenes_utils


class T4Dataset(NuScenesDataset):
    """T4Dataset Dataset base class

    The descriptions below are for the methods that aren't implemented this class.

    _format_bboxes:
    Convert the annotations to nuscenes-format and filtering the objects by the
    nuscenes-eval config.

    format_results:
    Apply _format_bboxes to the detection results.py
    """

    # This is default classes of training target and can be overwritten
    # by the config of `data.SPLIT.classes`.
    CLASSES = (
        "car",
        "truck",
        "bus",
        "bicycle",
        "pedestrian",
    )
    # To evaluate a model trained with waymo-open-dataset
    SimpleClassNameMapping = {
        "car": "car",
        "truck": "car",
        "bus": "car",
        "construction_vehicle": "car",
        "bicycle": "bicycle",
        "motorcycle": "bicycle",
        "pedestrian": "pedestrian",
    }

    # --------------------------------------------------------------------------
    # CAUTION!
    # You can't change EVAL_CLASS_RANGE after T4Dataset.__init__
    # These configs are loaded into NuScenesDataset in T4Dataset.__init__
    # --------------------------------------------------------------------------
    # These ranges are used for filtering the ground-truth and estimated objects for evaluation.
    # This filtering is applied in format_results() and _format_bboxes() functions.
    EVAL_CLASS_RANGE = {
        "car": 75,
        "truck": 75,
        "trailer": 75,
        "bus": 75,
        "construction_vehicle": 75,
        "bicycle": 75,
        "motorcycle": 75,
        "pedestrian": 75,
        "traffic_cone": 75,
        "barrier": 75,
    }

    # Some bicycles with rider have attribute of `without_rider`.
    bicycle_wrong_attribute_as_without_rider_scene_names = [
        "x2_gsm8_v1-0_nishishinjuku_3d_60f7768e709730003384fe20"
    ]

    def __init__(
        self,
        ignore_without_rider=False,
        target_location: Optional[str] = None,
        evaluate_model_trained_with_waymo: bool = False,
        eval_class_range: Optional[dict] = None,
        **kwargs,
    ):
        """

        Attributes:
        data_infos (list[dict]): the loaded pkl file created by create_data_info.
            Each dist should contains the following keys:
            - lidar_path (str): the path to the lidar data
            - scene_token (str): the scene token of t4dataset
            - token (str): the sample token of t4dataset
            - sweeps (list[dict]): the list of sweeps data
            - cams (dict[str, dict]): the camera data
            - lidar2ego_translation (np.ndarray):
            - lidar2ego_rotation (np.ndarray):
            - ego2global_translation (np.ndarray):
            - timestamp (int): the unix timestamp (μ sec)
            - gt_boxes (list[dict]): ground-truth boxes
            - gt_nusc_names (list[str]): class names of t4dataset format
            - gt_names (list[str]): class names mapped by NameMapping
            - gt_attrs (list[str]): object attributes
            - gt_velocity (np.ndarray): velocity of boxes
            - gt_scores (np.ndarray): (Optional) scores of boxes for pseudo labels.
            - num_lidar_pts (np.ndarray): number of lidar points in objects
            - num_radar_pts (np.ndarray): number of radar points in objects
            - valid_flag (np.ndarray): boolean value that the objects are valid
            - annotations_2d (dict[str, dict]):
        """

        super().__init__(**kwargs)

        # NOTE(yukke42): get_eval_class_range() must be before _get_nusc_eval_config().
        self.EVAL_CLASS_RANGE = self.get_eval_class_range(eval_class_range)
        self.eval_detection_configs = self._get_nusc_eval_config()
        self.evaluate_model_trained_with_waymo = evaluate_model_trained_with_waymo
        self.ignore_without_rider = ignore_without_rider

        # available locations
        # - lidar_cuboid_shiojiri_2hz
        # - lidar_cuboid_west_shinjuku_2hz
        # - lidar_cuboid_west_shinjuku_validation_10hz
        # - lidar_cuboid_odaiba_2hz
        # - x2_gsm8_v1-0_nishishinjuku_3d
        # - x2_gsm8_v1-0_glp-atsugi_3d
        if target_location is not None:
            data_info_size = len(self.data_infos)
            self.data_infos = list(
                filter(lambda info: info["location"] == target_location, self.data_infos)
            )
            print_log(
                f"data_infos is filtered by location name of `{target_location}`:"
                f" {data_info_size} -> {len(self.data_infos)}"
            )
            assert len(self.data_infos) > 0, f"No target location: {target_location}"

    @classmethod
    def get_eval_class_range(cls, eval_class_range: Optional[dict] = None):
        if eval_class_range is None:
            return cls.EVAL_CLASS_RANGE

        if isinstance(eval_class_range, dict):
            return eval_class_range
        else:
            raise TypeError(f"eval_class_range must be dict: {type(eval_class_range)}")

    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        # input_dict.update({"ann_info": self.get_ann_info(index)})
        return input_dict

    def get_ann_info(self, index: int):
        """Get annotation info according to the given index.

        Note (yukke42):
        The method is copied from
        https://github.com/open-mmlab/mmdetection3d/blob/v0.18.1/mmdet3d/datasets/nuscenes_dataset.py#L250-L297
        since we want to add new processes:
        - filtering objects by attributes
        - use SimpleClassNameMapping to evaluate the model trained on waymo by the t4dataset
        - if with_velocity is false, add velocity values for CenterPoint
        But, it may be better to implement them after `anns_results = super().get_data_info(index)`

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_list[index]["ann_info"]
        if self.use_valid_flag:
            mask = info["bbox_3d_isvalid"]
        else:
            mask = info["num_lidar_pts"] > 0

        if self.ignore_without_rider:
            mask_without_riders = np.array(["without_rider" not in x for x in info["gt_attrs"]])
            mask *= mask_without_riders

        gt_bboxes_3d = info["gt_bboxes_3d"][mask]  # x, y, z, w, l, h, yaw
        gt_names_3d = info["gt_nusc_name"][mask]
        gt_attrs_3d = info["gt_attrs"][mask]
        gt_scores_3d = info["gt_scores"][mask] if "gt_scores" in info.keys() else None
        instance_tokens = None
        if "instance_tokens" in info:
            instance_tokens = np.array(info["instance_tokens"])[mask]

        # new class of motorcycle and bicycle
        # in t4dataset, bicycle with and without rider are annotated as `bicycle``
        # if not any(
        #     [
        #         token in info["scene_name"]
        #         for token in self.bicycle_wrong_attribute_as_without_rider_scene_names
        #     ]
        # ):
        #     class_names = ["motorcycle", "bicycle"]
        #     without_rider_attrs = [
        #         "vehicle_state.parked",
        #         "cycle_state.without_rider",
        #         "motorcycle_state.without_rider",
        #     ]
        #     for i, (name, attr) in enumerate(zip(gt_names_3d, gt_attrs_3d)):
        #         if name in class_names and attr in without_rider_attrs:
        #             gt_names_3d[i] += "_without_rider"

        gt_labels_3d = []
        for cat in gt_names_3d:
            if self.evaluate_model_trained_with_waymo:
                cat = self.SimpleClassNameMapping.get(cat, "ignore")

            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["velocities"][mask]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d.numpy(), gt_velocity], axis=-1)
        else:
            # CenterPoint needs velocity
            gt_bboxes_3d = np.concatenate(
                [gt_bboxes_3d, np.zeros((gt_bboxes_3d.shape[0], 2))], axis=-1
            )

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5),
        ).convert_to(self.box_mode_3d)

        # 2d annotations
        gt_bboxes_2d, gt_camera_token, gt_labels_2d, img_filename = ([] for i in range(4))
        if "annos_2d" in info:
            sweeps = info["sweeps"]
            for i in range(len(sweeps) + 1):
                anno_2d = info["annos_2d"][i]
                img_filename.append(anno_2d["filename"])
                gt_bboxes_2d.append(anno_2d["bbox"])  # x1y1,x2y2
                gt_names_2d = anno_2d["name"]
                gt_camera_token.append(anno_2d["camera_token"])
                gt_label_2d = []
                for cat in gt_names_2d:
                    if cat in self.CLASSES:
                        gt_label_2d.append(self.CLASSES.index(cat))
                    else:
                        gt_label_2d.append(-1)
                gt_labels_2d.append(np.array(gt_label_2d).astype(np.int64))

        anns_results = dict(
            cams=info.get("cams", []),
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_scores_3d=gt_scores_3d,
            gt_names=gt_names_3d,
            instance_tokens=instance_tokens,
            img_filename=img_filename,
            bboxes=gt_bboxes_2d,
            labels=gt_labels_2d,
            gt_camera_token=gt_camera_token,
        )
        return anns_results

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process the `instances` field to
        `ann_info` in training stage.

        This function is modified to avoid hard-coded processes for nuscenes dataset.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        # to parse lidar info in this function
        use_lidar = self.modality["use_lidar"]
        self.modality["use_lidar"] = False
        info = super().parse_data_info(info)
        self.modality["use_lidar"] = use_lidar

        # modified from https://github.com/open-mmlab/mmdetection3d/blob/v1.2.0/mmdet3d/datasets/det3d_dataset.py#L279-L296
        if self.modality["use_lidar"]:
            info["lidar_points"]["lidar_path"] = osp.join(
                self.data_prefix.get("pts", ""), info["lidar_points"]["lidar_path"]
            )

            info["num_pts_feats"] = info["lidar_points"]["num_pts_feats"]
            info["lidar_path"] = info["lidar_points"]["lidar_path"]
            if "lidar_sweeps" in info:
                for sweep in info["lidar_sweeps"]:
                    # NOTE(yukke42): modified to avoid hard-coded processes for nuscenes dataset
                    # --- modification ---
                    file_suffix = sweep["lidar_points"]["lidar_path"]
                    # --- end ---
                    if "samples" in sweep["lidar_points"]["lidar_path"]:
                        sweep["lidar_points"]["lidar_path"] = osp.join(
                            self.data_prefix["pts"], file_suffix
                        )
                    else:
                        sweep["lidar_points"]["lidar_path"] = osp.join(
                            self.data_prefix["sweeps"], file_suffix
                        )

        return info

    def _get_nusc_eval_config(self):
        # Note: same as the default values except class_range.
        eval_config_dict = {
            "class_names": self.CLASSES,
            "class_range": self.EVAL_CLASS_RANGE,
            "dist_fcn": "center_distance",
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "max_boxes_per_sample": 500,
            "mean_ap_weight": 5,
        }
        return nuscenes_utils.DetectionConfig.deserialize(eval_config_dict)

    def _get_hdl_eval_config(
        self, class_name="car", hdl_pedestrian_matching=None, **eval_config_kwargs
    ):
        if hdl_pedestrian_matching is None:
            condition = {
                "criteria": "iou",
                "threshold": 0.5 if class_name in ["pedestrian", "bicycle"] else 0.7,
            }
        else:
            assert hdl_pedestrian_matching > 0 and hdl_pedestrian_matching < 10
            thresholds = {"pedestrian": hdl_pedestrian_matching, "bicycle": 0.5}
            condition = {
                "criteria": "iou" if class_name not in ["pedestrian"] else "distance",
                "threshold": thresholds[class_name] if class_name in thresholds else 0.7,
            }
        eval_config_dict = {
            "es_box_filter": f'detection_name == "{class_name}"' if class_name != "all" else "",
            "gt_box_filter": f'detection_name == "{class_name}"' if class_name != "all" else "",
            "overlap_condition": condition,
        }
        eval_config_dict.update(eval_config_kwargs)
        return hdl_utils.HDLEvaluationConfig.deserialize(eval_config_dict)

    def _format_gt_to_nusc(self, output_dir: str, pipeline: Optional[List[Dict]] = None):
        """Convert ground-truth annotations to nuscenes Box format.

        Args:
            output_dir (str): the path to output directory
            pipeline (list[dict], optional): pipeline for formatting GTs

        Returns:
            str: the path to the formatted ground-truth file
        """

        if pipeline is not None:
            pipeline = self._get_pipeline(pipeline)

        gt_dicts = []
        for sample_id in range(len(self.data_infos)):
            annos = self.get_ann_info(sample_id)
            mask = annos["gt_labels_3d"] != -1

            # Setup dict for pipeline
            input_dict = dict(
                gt_bboxes_3d=annos["gt_bboxes_3d"][mask],
                gt_labels_3d=torch.from_numpy(annos["gt_labels_3d"][mask]),
            )
            if pipeline is not None:
                input_dict = pipeline(input_dict)

            # We use differnet key names here to use _format_bbox()
            gt_dicts.append(
                dict(
                    boxes_3d=input_dict["gt_bboxes_3d"],
                    labels_3d=input_dict["gt_labels_3d"],
                    scores_3d=torch.ones(len(input_dict["gt_labels_3d"])),
                )
            )

        gt_path = self._format_bbox(
            gt_dicts,
            self.annotations_frame,
            jsonfile_prefix=osp.join(output_dir, "gt"),
        )

        return gt_path

    def _compose_scene_dicts_from_flat_dict(
        self, bboxes_dict: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Group bboxes by scenes

        Args:
            bboxes_dict (dict[str, any]): nuscenes-format bboxes of all scenes

        Returns:
            dict[dict[str, any]]: nuscenes-format bboxes for each scene
        """

        dict_scene = defaultdict(dict)
        sample_token_to_scene_token = {d["token"]: d["scene_token"] for d in self.data_infos}
        for sample_token, scene_token in sample_token_to_scene_token.items():
            dict_scene[scene_token].update({sample_token: bboxes_dict[sample_token]})
        return dict_scene

    def _evaluate_nuscenes_each_scene(
        self,
        output_dir: str,
        result_path: str,
        gt_path: str,
        logger: Optional[logging.Logger] = None,
    ):
        result_dict = mmengine.load(result_path)["results"]
        gt_dict = mmengine.load(gt_path)["results"]

        result_dict_scene = self._compose_scene_dicts_from_flat_dict(result_dict)
        gt_dict_scene = self._compose_scene_dicts_from_flat_dict(gt_dict)
        sample_token_to_scene_token = {d["token"]: d["scene_token"] for d in self.data_infos}

        ap_dicts: List[Dict] = list()
        scene_tokens: List[str] = list(sample_token_to_scene_token.values())
        scene_tokens: List[str] = sorted(set(scene_tokens), key=scene_tokens.index)
        for scene_token in scene_tokens:
            print_log(f"===== {scene_token} =====", logger=logger)
            nusc_eval = nuscenes_utils.nuScenesDetectionEval(
                config=self._get_nusc_eval_config(),
                result_boxes=result_dict_scene[scene_token],
                gt_boxes=gt_dict_scene[scene_token],
                meta=self.modality,
                eval_set="val",
                output_dir=output_dir,
                verbose=False,
            )
            metrics, _ = nusc_eval.evaluate()
            metrics_summary = metrics.serialize()

            metrics_str, ap_dict = nuscenes_utils.format_nuscenes_metrics(
                metrics_summary, sorted(set(self.CLASSES), key=self.CLASSES.index)
            )
            ap_dicts.append(ap_dict)

            print_log(metrics_str, logger=logger)

        ap_dict = {"mAP": sum([d["mAP"] for d in ap_dicts]) / len(ap_dicts)}
        print_log(f"mAP = {ap_dict['mAP']:.4f}", logger=logger)

        return ap_dict

    def _evaluate_nuscenes_all_scenes(
        self,
        output_dir: str,
        result_path: str,
        gt_path: str,
        logger: Optional[logging.Logger] = None,
    ):
        result_dict = mmengine.load(result_path)["results"]
        gt_dict = mmengine.load(gt_path)["results"]

        nusc_eval = nuscenes_utils.nuScenesDetectionEval(
            config=self._get_nusc_eval_config(),
            result_boxes=result_dict,
            gt_boxes=gt_dict,
            meta=self.modality,
            eval_set="val",
            output_dir=output_dir,
            verbose=False,
        )
        metrics, _ = nusc_eval.evaluate()
        metrics_summary = metrics.serialize()

        metrics_str, ap_dict = nuscenes_utils.format_nuscenes_metrics(
            metrics_summary, sorted(set(self.CLASSES), key=self.CLASSES.index)
        )

        scene_tokens = {d["scene_token"] for d in self.data_infos}
        print_log(f"===== {len(scene_tokens)} scenes ======", logger=logger)
        print_log(metrics_str, logger=logger)
        print_log(f"mAP = {ap_dict['mAP']:.4f}", logger=logger)
        return ap_dict

    def _evaluate_lyft_each_scene(
        self,
        output_dir: str,
        result_path: str,
        gt_path: str,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, float]:
        print_log("Start to evaluate with lyft Evaluator...", logger=logger)
        result_dict = mmengine.load(result_path)["results"]
        gt_dict = mmengine.load(gt_path)["results"]

        result_dict_scene = self._compose_scene_dicts_from_flat_dict(result_dict)
        gt_dict_scene = self._compose_scene_dicts_from_flat_dict(gt_dict)
        sample_token_to_scene_token = {d["token"]: d["scene_token"] for d in self.data_infos}

        scene_tokens: List[str] = list(sample_token_to_scene_token.values())
        scene_tokens: List[str] = sorted(set(scene_tokens), key=scene_tokens.index)
        for scene_token in scene_tokens:
            ap_dict: Dict = {}
            print_log(f"===== {scene_token} =====", logger=logger)
            lyft_utils.lyft_eval(
                nusc_det_annos=result_dict_scene[scene_token],
                nusc_gt_annos=gt_dict_scene[scene_token],
                class_names=self.CLASSES,
                output_dir=output_dir,
                logger=logger,
                metric="iou_bev",
            )

        return {}

    def _evaluate_lyft_all_scenes(
        self,
        output_dir: str,
        result_path: str,
        gt_path: str,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, float]:
        print_log("Start to evaluate with lyft Evaluator...", logger=logger)
        scene_tokens = {d["scene_token"] for d in self.data_infos}
        num_scenes = len(scene_tokens)
        print_log(f"===== {num_scenes} scenes =====", logger=logger)

        result_dict = mmengine.load(result_path)["results"]
        gt_dict = mmengine.load(gt_path)["results"]

        ap_dict = lyft_utils.lyft_eval(
            nusc_det_annos=result_dict,
            nusc_gt_annos=gt_dict,
            class_names=self.CLASSES,
            output_dir=output_dir,
            logger=logger,
            metric="iou_bev",
        )
        return ap_dict

    def _evaluate_hdl_each_scene(
        self,
        output_dir: str,
        result_path: str,
        gt_path: str,
        logger: Optional[logging.Logger] = None,
        hdl_decision_threshold: float = 0.0,
        hdl_pedestrian_matching: Optional[float] = None,
    ):
        """Evaluation using HDLEvaluator.

        Args:
            output_dir (str): A directory to save the evaluation results.
            result_path (str): Path of the result file loadable with `mmengine.load`.
            gt_path (str): Path of the ground-truth file loadable with `mmengine.load`.
            logger (logging.Logger, optional): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            (dict): Dictionary of the evaluation results

        """
        # Load detection results and ground-truth
        result_dict = mmengine.load(result_path)["results"]
        gt_dict = mmengine.load(gt_path)["results"]

        # Identify scenes
        result_scene_dict = self._compose_scene_dicts_from_flat_dict(result_dict)
        gt_scene_dict = self._compose_scene_dicts_from_flat_dict(gt_dict)

        # Make sure that all scenes matches between the detection result and ground-truth
        assert len(set(result_scene_dict.keys()).difference(set(gt_scene_dict.keys()))) == 0

        print_log("Start to evaluate with HDLEvaluator...", logger=logger)
        evaluation_scores = {}
        for scene in gt_scene_dict.keys():
            scene_scores = {}
            prog_bar = mmengine.ProgressBar(len(self.CLASSES))
            for class_name in list(self.CLASSES) + ["all"]:
                for f in result_scene_dict[scene].keys():
                    resp = [
                        r
                        for r in result_scene_dict[scene][f]
                        if r["detection_score"] > hdl_decision_threshold
                    ]
                    result_scene_dict[scene][f] = resp
                evaluator = hdl_utils.HDLEvaluator(
                    config=self._get_hdl_eval_config(
                        class_name=class_name, hdl_pedestrian_matching=hdl_pedestrian_matching
                    ),
                    es_boxes=result_scene_dict[scene],
                    gt_boxes=gt_scene_dict[scene],
                )
                scores = evaluator.evaluate()
                scene_scores[class_name] = scores
                prog_bar.update()
            df = pd.DataFrame.from_dict(scene_scores, orient="index")
            df.drop(columns=["evaluation_conditions"], inplace=True)
            df.index.name = "class_name"
            print_log(f"===== {scene} =====", logger=logger)
            print_log(
                "---------------- HDLEvaluator results-----------------\n"
                + df.to_markdown(mode="str"),
                logger=logger,
            )
            evaluation_scores[scene] = scene_scores

        # Summarize the scores
        summary = {}
        for class_name in list(self.CLASSES) + ["all"]:
            scores = [scene_scores[class_name] for scene_scores in evaluation_scores.values()]
            scores_df = pd.DataFrame.from_records(scores)
            means = scores_df[hdl_utils.HDLEvaluator.summarizable_criterion].mean().to_dict()
            summary.update({f"{class_name}_{k}": v for k, v in means.items()})

        # Save the evaluation result
        evaluation_scores["summary"] = summary
        output_file = osp.join(output_dir, "hdl_evaluator_results.pkl")
        mmengine.dump(evaluation_scores, output_file)

        return summary

    def _evaluate_hdl_all_scenes(
        self,
        output_dir: str,
        result_path: str,
        gt_path: str,
        logger: Optional[logging.Logger] = None,
        show_markdown: bool = True,
        hdl_decision_threshold: float = 0.0,
        hdl_pedestrian_matching: Optional[float] = None,
        **eval_config_kwargs,
    ):
        """Evaluation using HDLEvaluator.

        Args:
            output_dir (str): A directory to save the evaluation results.
            result_path (str): Path of the result file loadable with `mmengine.load`.
            gt_path (str): Path of the ground-truth file loadable with `mmengine.load`.
            logger (logging.Logger, optional): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            (dict): Dictionary of the evaluation results

        """
        # Load detection results and ground-truth
        result_dict = mmengine.load(result_path)["results"]
        gt_dict = mmengine.load(gt_path)["results"]

        scene_tokens = {d["scene_token"] for d in self.data_infos}
        num_scenes = len(scene_tokens)

        print_log("Start to evaluate with HDLEvaluator...", logger=logger)
        confusion_matrix = pd.DataFrame()
        evaluation_scores = {}
        for class_name in list(self.CLASSES) + ["all"]:
            for f in result_dict.keys():
                resp = [r for r in result_dict[f] if r["detection_score"] > hdl_decision_threshold]
                result_dict[f] = resp
            evaluator = hdl_utils.HDLEvaluator(
                config=self._get_hdl_eval_config(
                    class_name=class_name,
                    hdl_pedestrian_matching=hdl_pedestrian_matching,
                    **eval_config_kwargs,
                ),
                es_boxes=result_dict,
                gt_boxes=gt_dict,
            )
            scores = evaluator.evaluate()
            evaluation_scores[class_name] = scores

            # confidence metrics
            confidences = sum(
                [boxes.score.cpu().tolist() for boxes in evaluator.es_boxes.values()], []
            )
            if len(confidences) > 0:
                true_positives = evaluator.es_true_positives
                cc_vis.add_plots(true_positives, confidences, label=class_name)

            # confusion matrix
            if class_name == "all":
                confusion_matrix = evaluator.get_confusion_matrix(
                    labels=sorted(set(self.CLASSES), key=self.CLASSES.index)
                )


        df = pd.DataFrame.from_dict(evaluation_scores, orient="index")
        df.drop(columns=["evaluation_conditions"], inplace=True)
        df.index.name = "class_name"
        if show_markdown:
            print_log(f"===== {num_scenes} scenes =====", logger=logger)
            print_log(
                "---------------- HDLEvaluator results-----------------\n"
                + df.to_markdown(mode="str"),
                logger=logger,
            )
            print_log(
                "---------------- Confusion matrix-----------------\n"
                + confusion_matrix.to_markdown(mode="str"),
                logger=logger,
            )

        # Summarize the scores
        summary = {}
        for class_name in list(self.CLASSES) + ["all"]:
            scores_df = pd.DataFrame.from_records(evaluation_scores[class_name])
            scores = scores_df[hdl_utils.HDLEvaluator.summarizable_criterion].mean().to_dict()
            summary.update({f"{class_name}_{k}": v for k, v in scores.items()})

        # Save the evaluation result
        output_file = osp.join(output_dir, "hdl_evaluator_all_results.pkl")
        mmengine.dump(summary, output_file)

        return summary

    def _evaluate_single(
        self,
        result_path: str,
        logger: Optional[logging.Logger] = None,
        metric: Union[str, List[str]] = "nuscenes",
        result_name: str = "pts_bbox",
        pipeline_for_gt: Optional[List[Dict]] = None,
        hdl_decision_threshold: float = 0.0,
        hdl_pedestrian_matching: Optional[float] = None,
    ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str | list[str]): Metrics to be evaluated. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])
        gt_path = self._format_gt_to_nusc(output_dir, pipeline=pipeline_for_gt)

        if self.data_class_mapping:
            # edit class names in dataset, mapping by data_class_mapping
            gt_dict = mmengine.load(gt_path)
            map_dataset_classes(gt_dict["results"], self.data_class_mapping)
            mmengine.dump(gt_dict, gt_path)

        ap_dict = dict()
        if "nuscenes-each-scene" in metric:
            summary = self._evaluate_nuscenes_each_scene(output_dir, result_path, gt_path, logger)
        if "nuscenes" in metric:
            summary = self._evaluate_nuscenes_all_scenes(output_dir, result_path, gt_path, logger)
            ap_dict.update({f"nuScenes_{k}": v for k, v in summary.items()})
        if "hdl-each-scene" in metric:
            summary = self._evaluate_hdl_each_scene(
                output_dir,
                result_path,
                gt_path,
                logger,
                hdl_decision_threshold=hdl_decision_threshold,
                hdl_pedestrian_matching=hdl_pedestrian_matching,
            )
            ap_dict.update({f"HDL(scene)/{k}": v for k, v in summary.items()})
        if "hdl" in metric:
            summary = self._evaluate_hdl_all_scenes(
                output_dir,
                result_path,
                gt_path,
                logger,
                hdl_decision_threshold=hdl_decision_threshold,
                hdl_pedestrian_matching=hdl_pedestrian_matching,
            )
            ap_dict.update({f"HDL_{k}": v for k, v in summary.items()})
        if "lyft-each-scene" in metric:
            summary = self._evaluate_lyft_each_scene(output_dir, result_path, gt_path, logger)
        if "lyft" in metric:
            summary = self._evaluate_lyft_all_scenes(output_dir, result_path, gt_path, logger)
            ap_dict.update({f"lyft_{k}": v for k, v in summary.items()})

        return ap_dict

    def evaluate(
        self,
        results,
        metric: Union[str, List[str]] = ["nuscenes", "hdl", "hard_decision"],
        logger: Optional[logging.Logger] = None,
        jsonfile_prefix: Optional[str] = None,
        result_names: List[str] = ["pts_bbox"],
        show: bool = False,
        out_dir: Optional[str] = None,
        pipeline_for_gt: Optional[List[Dict]] = None,
        pipeline: Optional[List[Dict]] = None,
        data_class_mapping: Optional[Dict[str, str]] = None,
        hard_decision_betas: List[float] = [1.0],
        hard_decision_thresholds: List[Optional[float]] = [None],
        hdl_decision_threshold: float = 0.0,
        hdl_pedestrian_matching: Optional[float] = None,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: ["nuscenes", "hdl", "hard_decision"],
            logger (logging.Logger, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize. Default: False.
            out_dir (str, optional): Path to save the vis and hard decision results.
                Default: None.
            pipeline_for_gt (list[dict], optional): Pipeline list for formatting GTs.
                Default: None.
            pipeline (list[dict], optional): Pipeline list for showing.
                Default: None.
            data_class_mapping (dict, optional): Dict used for mapping class names in dataset.
                Default: None.
            hard_decision_betas (List[float]): Beta values for f-value used in hard decision.
                Default: [1.0, 2.0]
            hard_decision_thresholds (List[Optional[float]]): Thresholds used in hard decision.
                Default: [None, 0.4]
            hdl_decision_threshold (Optional[float]): Threshold used in hdl
                Default: 0.0
            hdl_pedestrian_matching: Optional[float]: Distance for pedestrians, if None, use IoU
                Default: None

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        # naive workaround for the case that the model does not estimate velocity
        # reference: https://github.com/open-mmlab/mmdetection3d/issues/292
        for _result in results:
            result = _result["pts_bbox"] if "pts_bbox" in _result.keys() else _result
            if result["boxes_3d"].tensor.shape[1] == 7:
                result["boxes_3d"].tensor = torch.cat(
                    (
                        result["boxes_3d"].tensor,
                        torch.zeros(result["boxes_3d"].tensor.shape[0], 2),
                    ),
                    dim=1,
                )

        try:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        except AssertionError:
            warnings.warn("Model outputs contains invalid values. Skipped evaluation.")
            return {}

        print_log(
            f"EVAL_CLASS_RANGE:\n{json.dumps(self.EVAL_CLASS_RANGE, sort_keys=True, indent=2)}",
            logger=logger,
        )

        # soft decision eval
        self.data_class_mapping: dict = data_class_mapping
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print("Evaluating bboxes of {}".format(name))
                ret_dict = self._evaluate_single(
                    result_files[name],
                    logger=logger,
                    metric=metric,
                    pipeline_for_gt=pipeline_for_gt,
                    hdl_decision_threshold=hdl_decision_threshold,
                    hdl_pedestrian_matching=hdl_pedestrian_matching,
                )
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(
                result_files,
                logger=logger,
                metric=metric,
                pipeline_for_gt=pipeline_for_gt,
                hdl_decision_threshold=hdl_decision_threshold,
                hdl_pedestrian_matching=hdl_pedestrian_matching,
            )

        if "hard_decision" in metric:
            # hard decision eval
            if out_dir is None:
                out_dir = tmp_dir.name

            if data_class_mapping is not None:
                warnings.warn("data_class_mapping is not support in hard decision eval.")

            for thres, beta in itertools.product(hard_decision_thresholds, hard_decision_betas):
                out_csv_name = os.path.join(
                    out_dir,
                    f"hard_decision_results_thres{str(thres).lower()}_beta{str(beta)}.csv",
                )
                self._evaluate_hard_decision(
                    results,
                    jsonfile_prefix,
                    logger,
                    hard_decision_threshold=thres,
                    f1_value_beta=beta,
                    out_csv_name=out_csv_name,
                    pipeline_for_gt=pipeline_for_gt,
                )

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def _evaluate_hard_decision(
        self,
        results: List[Dict],
        jsonfile_prefix: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        detection_ranges: List[Union[int, List[int]]] = [
            [0, 25],
            [25, 50],
            [50, 75],
        ],
        hard_decision_threshold: Optional[float] = 0.4,
        f1_value_beta: float = 1.0,
        overlap_conditions: List[Dict] = [
            {
                "criteria": "iou",
                "threshold": 0.3,
            },
            {
                "criteria": "iou",
                "threshold": 0.5,
            },
            {
                "criteria": "iou",
                "threshold": 0.7,
            },
        ],
        metrics: List[str] = [
            "num-gt-boxes",
            "AP",
            "best-f-value",
            "best-f-value-precision",
            "best-f-value-recall",
            "best-f-value-threshold",
            "APH",
            "best-f-value-h",
            "best-f-value-precision-h",
            "best-f-value-recall-h",
            "best-f-value-threshold-h",
        ],
        out_csv_name: Optional[str] = None,
        pipeline_for_gt: Optional[List[Dict]] = None,
    ):
        default_class_range_dict = self.eval_detection_configs.class_range.copy()
        result_dict = defaultdict(list)

        # range-wise evaluation
        for detection_range in detection_ranges:
            class_range_dict = default_class_range_dict.copy()
            # overwrite default values
            for k, v in class_range_dict.items():
                if v != 0:
                    class_range_dict[k] = detection_range
            self.eval_detection_configs.class_range.update(class_range_dict)
            result_path_dict, tmp_dir = self.format_results(results, jsonfile_prefix)
            assert "pts_bbox" in result_path_dict
            result_path = result_path_dict["pts_bbox"]
            output_dir = osp.join(*osp.split(result_path)[:-1])
            gt_path = self._format_gt_to_nusc(output_dir, pipeline=pipeline_for_gt)

            for overlap_condition in overlap_conditions:
                summary = self._evaluate_hdl_all_scenes(
                    output_dir=output_dir,
                    result_path=result_path,
                    gt_path=gt_path,
                    logger=logger,
                    show_markdown=False,
                    overlap_condition=overlap_condition,
                    hard_decision_threshold=hard_decision_threshold,
                    f1_value_beta=f1_value_beta,
                )
                for c in list(self.CLASSES) + ["all"]:
                    min, max = detection_range
                    result_dict["detection_range"] += [f"{min}-{max} [m]"]
                    result_dict["overlap_condition"] += [
                        f"{overlap_condition['criteria']}@{overlap_condition['threshold']}"
                    ]
                    result_dict["class"] += [c]
                    for m in metrics:
                        result_dict[m] += [summary[f"{c}_{m}"]]

            if tmp_dir is not None:
                tmp_dir.cleanup()

        # convert to dataframe and calculate additional metrics
        df = pd.DataFrame(result_dict)
        mean_result_dict = defaultdict(list)

        classes_wo_all_ = list(self.CLASSES)
        classes_ = classes_wo_all_ + ["all"]
        detection_ranges_ = df["detection_range"].unique().tolist()
        overlap_conditions_ = df["overlap_condition"].unique().tolist()

        # calculate class-wise averaged
        for dr in detection_ranges_:
            for oc in overlap_conditions_:
                mean_result_dict["class"] += ["avg (except all)"]
                mean_result_dict["detection_range"] += [dr]
                mean_result_dict["overlap_condition"] += [oc]
                for m in metrics:
                    mean_result_dict[m] += [
                        df[
                            (df["detection_range"] == dr)
                            & (df["overlap_condition"] == oc)
                            # except "all" class
                            & df["class"].isin(classes_wo_all_)
                        ][m].mean()
                    ]

        # calculate detection range averaged
        for c in classes_:
            for oc in overlap_conditions_:
                mean_result_dict["class"] += [c]
                mean_result_dict["overlap_condition"] += [oc]
                mean_result_dict["detection_range"] += ["avg"]
                for m in metrics:
                    mean_result_dict[m] += [
                        df[
                            (df["class"] == c)
                            & (df["overlap_condition"] == oc)
                            & (df["detection_range"].isin(detection_ranges_))
                        ][m].mean()
                    ]

        # calculate overlap-condition averaged
        for c in classes_:
            for dr in detection_ranges_:
                mean_result_dict["class"] += [c]
                mean_result_dict["detection_range"] += [dr]
                mean_result_dict["overlap_condition"] += ["avg"]
                for m in metrics:
                    mean_result_dict[m] += [
                        df[
                            (df["detection_range"] == dr)
                            & (df["class"] == c)
                            & (df["overlap_condition"].isin(overlap_conditions_))
                        ][m].mean()
                    ]

        # calculate detection range and overlap condition averaged
        for c in classes_:
            mean_result_dict["class"] += [c]
            mean_result_dict["overlap_condition"] += ["avg"]
            mean_result_dict["detection_range"] += ["avg"]
            for m in metrics:
                mean_result_dict[m] += [
                    df[
                        (df["class"] == c)
                        & (df["detection_range"].isin(detection_ranges_))
                        & (df["overlap_condition"].isin(overlap_conditions_))
                    ][m].mean()
                ]

        # calculate all averaged
        mean_result_dict["class"] += ["avg (except all)"]
        mean_result_dict["detection_range"] += ["avg"]
        mean_result_dict["overlap_condition"] += ["avg"]
        for m in metrics:
            mean_result_dict[m] += [
                df[
                    (df["detection_range"].isin(detection_ranges_))
                    & (df["overlap_condition"].isin(overlap_conditions_))
                    & (df["class"].isin(classes_wo_all_))
                ][m].mean()
            ]
        df = pd.concat([df, pd.DataFrame(mean_result_dict)], axis=0)

        # print
        df = (
            df.set_index(["overlap_condition", "class", "detection_range"], drop=True)
            .sort_index()
            .sort_index(axis=1, ascending=False)
            .reindex(overlap_conditions_ + ["avg"], level="overlap_condition")
            .reindex(classes_ + ["avg (except all)"], level="class")
            .reindex(detection_ranges_ + ["avg"], level=2)
        )
        print_log(
            f"---------------- HARD DECISION RESULTS (hard_decision_threshold={hard_decision_threshold},f1_value_beta={f1_value_beta})-----------------\n"
            + df.to_markdown(mode="str"),
            logger=logger,
        )
        if out_csv_name is not None:
            out_dir = os.path.dirname(out_csv_name)
            if len(out_dir) != 0:
                os.makedirs(out_dir, exist_ok=True)
            df.to_csv(out_csv_name, float_format="%.4f")

        # restore to default setting
        self.eval_detection_configs.class_range.update(default_class_range_dict)

        return result_dict

    def _format_bbox(self, results, bbox_frame=None, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            bbox_frame (str): The frame of the bboxes. This needed to be included since some databases generate bboxes in the lidar frame, whereas others do so in the ego frame.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.

        To use extended `lidar_nusc_box_to_global`, this method is just copied from:
        https://github.com/open-mmlab/mmdetection3d/blob/9556958fe1c6fe432d55a9f98781b8fdd90f4e9c/mmdet3d/datasets/nuscenes_dataset.py#L301

        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.eval_detection_configs,
                self.eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
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
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

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
        print("Results written to", res_path)
        mmengine.dump(nusc_submissions, res_path)
        return res_path


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    eval_version="detection_cvpr_2019",
):
    """Convert the box from ego to global coordinate.

    To filter objects located within the range, extended to accept the tuple or list in class_rage.
    The original code can be found in:
    https://github.com/open-mmlab/mmdetection3d/blob/9556958fe1c6fe432d55a9f98781b8fdd90f4e9c/mmdet3d/datasets/nuscenes_dataset.py#L622-L657

    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        # NOTE(kan-bayashi): Extend to accept list or tuple
        if isinstance(det_range, (list, tuple)):
            det_range_min, det_range_max = det_range
            if radius <= det_range_min or radius > det_range_max:
                continue
        else:
            if radius > det_range:
                continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list


if __name__ == "__main__":
    import argparse

    from awml_det3d.detection_3d.datasets import build_dataset
    from awml_det3d.detection_3d.tools.utils import load_base_config
    from mmengine import DictAction
    from mmengine.fileio.io import load

    parser = argparse.ArgumentParser(
        description="Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="config file path",
    )
    parser.add_argument(
        "result_path",
        help="config file path",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help=(
            "evaluation metrics, which depends on the dataset, e.g., 'bbox', "
            "'segm', 'proposal' for COCO, and 'mAP', 'recall' for PASCAL VOC"
        ),
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help=(
            "override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file. If the value to "
            'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
            "Note that the quotation marks are necessary and that no white space "
            "is allowed."
        ),
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help=(
            "custom options for evaluation, the key-value pair in xxx=yyy "
            "format will be kwargs for dataset.evaluate() function"
        ),
    )
    args = parser.parse_args()

    config = load_base_config(args.config, args.cfg_options)
    dataset = build_dataset(config.data.test)
    outputs = load(args.result_path)

    kwargs = {} if args.eval_options is None else args.eval_options
    eval_kwargs = config.get("evaluation", {}).copy()
    # hard-code way to remove EvalHook args
    for key in ["interval", "tmpdir", "startdir", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval, **kwargs))
    print(dataset.evaluate(outputs, **eval_kwargs))
