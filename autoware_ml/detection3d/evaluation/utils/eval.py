import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tqdm
from mmcv.transforms import BaseTransform
from mmengine.logging import MMLogger, print_log
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBox, EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.nuscenes import Box
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion


class BaseNameMapping(BaseTransform):
    # TODO(boczekbartek): handle 2D annotations
    MAPPING = dict()

    def __init__(self, classes: List[str], unk_label: int = -1):
        super().__init__()
        self.classes = classes
        self.unk_label = unk_label

    @staticmethod
    def encode_label(cat: str, classes: List[str], unk_label: int = -1) -> int:
        return classes.index(cat) if cat in classes else unk_label

    @staticmethod
    def map_classes(
        source_class_names: Sequence[str],
        target_class_names: List[str],
        mapping: Dict[str, str],
        unk_label: int = -1,
    ) -> Tuple[List[str], List[int]]:
        """Map classes from source to target classes.

        Args:
            source_class_names (list[str]): source classes to map
            target_class_names (list[str]): target classes to encode target labels. Order is important and will influence labels.
            mapping (dict): mapping from source to target classes
        """
        new_names: List[str] = []
        new_labels: List[int] = []

        for cat in source_class_names:
            new_cat = mapping.get(cat, cat)
            new_names.append(new_cat)
            new_labels.append(BaseNameMapping.encode_label(new_cat, target_class_names, unk_label))

        return new_names, new_labels

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        gt_names_3d: List[str] = results["ann_info"]["gt_names_3d"]
        gt_names_3d, gt_labels_3d = self.map_classes(
            gt_names_3d, self.classes, self.MAPPING, self.unk_label
        )
        results["gt_labels_3d"] = np.array(gt_labels_3d, dtype=np.int64)
        results["gt_names_3d"] = gt_names_3d
        return results

    def map_name(self, name: str, *args, **kwargs) -> str:
        return self.MAPPING.get(name, name)

class T4Box(DetectionBox):
    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_translation: Tuple[float, float, float] = (
            0,
            0,
            0,
        ),  # Translation to ego vehicle in meters.
        num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
        detection_name: str = "car",  # The class name used in the detection challenge.
        detection_score: float = -1.0,  # GT samples do not have a score.
        attribute_name: str = "",
    ):  # Box attribute. Each box can have at most 1 attribute.
        EvalBox.__init__(
            self, sample_token, translation, size, rotation, velocity, ego_translation, num_pts
        )  # Call the grandparents' __init__ method to avoid unnecessary checks in DetectionBox
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name


class T4DetectionConfig(DetectionConfig):
    """Data class that specifies the detection evaluation settings."""

    def __init__(
        self,
        class_names: List[str],
        class_range: Dict[str, int],
        dist_fcn: str,
        dist_ths: List[float],
        dist_th_tp: float,
        min_recall: float,
        min_precision: float,
        max_boxes_per_sample: int,
        mean_ap_weight: int,
    ):
        assert class_range.keys() == set(
            class_names
        ), "class_range must have keys for all classes."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight
        self.class_names = class_names

    @classmethod
    def deserialize(cls, content: dict):
        """Initialize from serialized dictionary."""
        return cls(
            content["class_names"],
            content["class_range"],
            content["dist_fcn"],
            content["dist_ths"],
            content["dist_th_tp"],
            content["min_recall"],
            content["min_precision"],
            content["max_boxes_per_sample"],
            content["mean_ap_weight"],
        )


class T4DetectionEvaluation(DetectionEval):
    def __init__(
        self,
        config: DetectionConfig,
        result_path: str,
        scene: str,
        output_dir: str,
        ground_truth_boxes: EvalBoxes,
        prediction_boxes: EvalBoxes,
        verbose: bool = True,
    ):
        """
        Initialize a DetectionEval object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param scene: The scene token to evaluate on
        :param output_dir: Folder to save plots and results to.
        :param ground_truth_boxes: The ground truth boxes.
        :param prediction_boxes: The predicted boxes.
        :param verbose: Whether to print to stdout.
        """
        self.result_path = result_path
        self.scene = scene
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.gt_boxes = ground_truth_boxes
        self.pred_boxes = prediction_boxes

        # Check result file exists.
        assert os.path.exists(result_path), "Error: The result file does not exist!"

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print("Initializing nuScenes detection evaluation")
        assert set(self.pred_boxes.sample_tokens) == set(
            self.gt_boxes.sample_tokens
        ), "Samples in split doesn't match samples in predictions."

        self.sample_tokens = self.gt_boxes.sample_tokens

    def run_and_save_eval(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :return: A dict that stores the high-level metrics and meta data.
        """
        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Dump the metric data, meta and metrics to disk.
        print("Saving metrics to: %s" % self.output_dir)
        metrics_summary = metrics.serialize()
        # metrics_summary["meta"] = self.meta.copy()
        with open(os.path.join(self.output_dir, "metrics_summary.json"), "w") as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, "metrics_details.json"), "w") as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        mean_AP = "{:.3g}".format(metrics_summary["mean_ap"])
        header = [
            "class_name",
            "mAP",
            "AP@0.5m",
            "AP@1.0m",
            "AP@2.0m",
            "AP@4.0m",
            #"error@trans_err",
            #"error@scale_err",
            #"error@orient_err",
            #"error@vel_err",
            #"error@attr_err",
        ]
        data = []
        class_aps = metrics_summary["mean_dist_aps"]
        class_tps = metrics_summary["label_tp_errors"]
        label_aps = metrics_summary["label_aps"]
        for class_name in class_aps.keys():
            data.append(
                [
                    class_name,
                    "{:.1f}".format(class_aps[class_name] * 100.0),
                    "{:.1f}".format(label_aps[class_name][0.5] * 100.0),
                    "{:.1f}".format(label_aps[class_name][1.0] * 100.0),
                    "{:.1f}".format(label_aps[class_name][2.0] * 100.0),
                    "{:.1f}".format(label_aps[class_name][4.0] * 100.0),
                    #"{:.3g}".format(class_tps[class_name]["trans_err"]),
                    #"{:.3g}".format(class_tps[class_name]["scale_err"]),
                    #"{:.3g}".format(class_tps[class_name]["orient_err"]),
                    #"{:.3g}".format(class_tps[class_name]["vel_err"]),
                    #"{:.3g}".format(class_tps[class_name]["attr_err"]),
                ]
            )
        metrics_table = {
            "header": header,
            "data": data,
            "total_mAP": mean_AP,
        }
        return metrics_summary, metrics_table


def print_metrics_table(
    header: List[str],
    data: List[List[str]],
    total_mAP: str = "",
    metric_name: str = "",
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
    header_str = (
        "| " + " | ".join(header[i].ljust(col_widths[i]) for i in range(len(header))) + " |\n"
    )
    # Format table_middle
    table_middle_str = "|" + " ---- |" * len(header) + "\n"

    # Format data rows
    rows = []
    for row in data:
        row_str = (
            "| "
            + " | ".join("{{:<{}}}".format(col_widths[i]).format(row[i]) for i in range(len(row)))
            + " |\n"
        )
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


# modified version from https://github.com/nutonomy/nuscenes-devkit/blob/9b165b1018a64623b65c17b64f3c9dd746040f36/python-sdk/nuscenes/eval/common/loaders.py#L53
# adds name mapping capabilities
def t4metric_load_gt(
    nusc: NuScenes,
    config: DetectionConfig,
    scene: str,
    verbose: bool = False,
    #name_mapping: Optional[BaseNameMapping] = None,
    name_mapping = None,
    post_mapping_dict: Optional[Dict[str, str]] = None,
    filter_attributions = [["vehicle.bicycle", "vehicle_state.parked"], ["vehicle.motorcycle", "vehicle_state.parked"]]
) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param config: The detection config.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param verbose: Whether to print messages to stdout.
    :param name_mapping: Map detection names loaded from GT, to the names used in the evaluation. Optional
    :param post_mapping_dict: A dictionary to map detection names after the name mapping. Optional
    :return: The GT boxes.
    """
    if verbose:
        print(
            "Loading annotations for {} split from nuScenes version: {}".format(
                scene, nusc.version
            )
        )
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s["token"] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    for sample_token in tqdm.tqdm(sample_tokens_all, leave=verbose):
        sample = nusc.get("sample", sample_token)
        sample_annotation_tokens = sample["anns"]

        scene_record = nusc.get("scene", sample["scene_token"])
        scene_name = scene_record["name"]

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:
            sample_annotation = nusc.get("sample_annotation", sample_annotation_token)
            detection_name = sample_annotation["category_name"]

            # Get attribute_name.
            try:
                attribute_name = get_attr_name(nusc, sample_annotation)
            except ValueError:
                attribute_name = ""

            # if name_mapping:
            #     #detection_name = name_mapping.map_name(detection_name, attribute_name, scene_name)

            if filter_attributions:
                is_filter = False

                for filter_attribution in filter_attributions:
                    if detection_name == filter_attribution[0] and attribute_name == filter_attribution[1]:
                        is_filter = True
                if is_filter is True:
                    continue

            if post_mapping_dict:
                detection_name = post_mapping_dict.get(detection_name, detection_name)

            if detection_name not in config.class_names:
                continue

            sample_boxes.append(
                T4Box(
                    sample_token=sample_token,
                    translation=sample_annotation["translation"],
                    size=sample_annotation["size"],
                    rotation=sample_annotation["rotation"],
                    velocity=nusc.box_velocity(sample_annotation["token"])[:2],
                    num_pts=sample_annotation["num_lidar_pts"]
                    + sample_annotation["num_radar_pts"],
                    detection_name=detection_name,
                    detection_score=-1.0,  # GT samples do not have a score.
                )
            )
        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print(
            "Loaded ground truth annotations for {} samples.".format(
                len(all_annotations.sample_tokens)
            )
        )

    # Add center distances.
    all_annotations = add_center_dist(nusc, all_annotations)

    if verbose:
        print("Filtering ground truth annotations")
    all_annotations = filter_eval_boxes(nusc, all_annotations, config.class_range, verbose=verbose)
    return all_annotations


# modified version of https://github.com/nutonomy/nuscenes-devkit/blob/9b165b1018a64623b65c17b64f3c9dd746040f36/python-sdk/nuscenes/eval/common/loaders.py#L21
# adds name mapping capabilities
def t4metric_load_prediction(
    nusc: NuScenes,
    config: DetectionConfig,
    scene: str,
    result_path: str,
    max_boxes_per_sample: int,
    verbose: bool = True,
    name_mapping: Optional[BaseNameMapping] = None,
    post_mapping_dict: Optional[Dict[str, str]] = None,
) -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param nusc: A NuScenes instance.
    :param config: The detection config.
    :param scene: The scene token to evaluate on.
    :param result_path: Path to the result file.
    :param max_boxes_per_sample: The maximum number of boxes per sample.
    :param verbose: Whether to print messages to stdout. Optional
    :param name_mapping: Map detection names loaded from GT, to the names used in the evaluation. Optional
    :param post_mapping_dict: A dictionary to map detection names after the name mapping. Optional
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert "results" in data, (
        "Error: No field `results` in result file. Please note that the result format changed."
        "See https://www.nuscenes.org/object-detection for more information."
    )

    data = map_detection_names(data, name_mapping, post_mapping_dict)

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data["results"], T4Box)
    meta = data["meta"]
    if verbose:
        print(
            "Loaded results from {}. Found detections for {} samples.".format(
                result_path, len(all_results.sample_tokens)
            )
        )

    all_results = filter_by_known_tokens(nusc, all_results)

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, (
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample
        )

    all_results = add_center_dist(nusc, all_results)

    # Filter boxes (distance, points per box, etc.).
    all_results = filter_eval_boxes(nusc, all_results, config.class_range, verbose=verbose)

    return all_results, meta


def map_detection_names(
    boxes: dict,
    name_mapping: Optional[BaseNameMapping],
    post_mapping_dict: Optional[Dict[str, str]],
) -> dict:
    """
    Maps detection names to the names used in the evaluation.
    :param boxes: The boxes to map.
    :param name_mapping: The name mapping object. Optional
    :param post_mapping_dict: A dictionary to map detection names after the name mapping. Optional
    :return: The mapped boxes.
    """

    if name_mapping is not None:
        for sample_token in boxes["results"]:
            for box in boxes["results"][sample_token]:
                if "detection_name" in box:
                    if "attribute_name" in box:
                        detection_name = name_mapping.map_name(
                            box["detection_name"], box["attribute_name"]
                        )
                        if post_mapping_dict:
                            detection_name = post_mapping_dict.get(detection_name, detection_name)
                        box["detection_name"] = detection_name
                    else:
                        detection_name = name_mapping.map_name(box["detection_name"], "")
                        if post_mapping_dict:
                            detection_name = post_mapping_dict.get(detection_name, detection_name)
                        box["detection_name"] = name_mapping.map_name(box["detection_name"], "")
    return boxes


def filter_by_known_tokens(nusc: NuScenes, eval_boxes: EvalBoxes) -> EvalBoxes:
    """
    Filters the boxes to only include those that are in the DB.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes filtered to only include boxes that are in the DB.
    """
    # Get all sample tokens in the DB.
    sample_tokens_all = [s["token"] for s in nusc.sample]

    # Filter boxes.
    for sample_token in eval_boxes.sample_tokens:
        if sample_token not in sample_tokens_all:
            eval_boxes.boxes.pop(sample_token)

    return eval_boxes


# modified version of https://github.com/nutonomy/nuscenes-devkit/blob/9b165b1018a64623b65c17b64f3c9dd746040f36/python-sdk/nuscenes/eval/common/loaders.py#L180
# we take into account the fact that the sample may have LIDAR_TOP or LIDAR_CONCAT
# also removes TrackingBox in favour of T4Box
def add_center_dist(nusc: NuScenes, eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get("sample", sample_token)
        if "LIDAR_TOP" in sample_rec["data"]:
            sd_record = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        elif "LIDAR_CONCAT" in sample_rec["data"]:
            sd_record = nusc.get("sample_data", sample_rec["data"]["LIDAR_CONCAT"])
        else:
            raise Exception(
                "Error: LIDAR data not available for sample! Expected either LIDAR_TOP or LIDAR_CONCAT."
            )
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (
                box.translation[0] - pose_record["translation"][0],
                box.translation[1] - pose_record["translation"][1],
                box.translation[2] - pose_record["translation"][2],
            )
            if isinstance(box, DetectionBox) or isinstance(box, T4Box):
                box.ego_translation = ego_translation
            else:
                raise NotImplementedError

    return eval_boxes


# modified from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/loaders.py#L207
# removes dependancy on a private function _get_box_class_field and TrackingBox
def filter_eval_boxes(
    nusc: NuScenes,
    eval_boxes: EvalBoxes,
    max_dist: Dict[str, float],
    verbose: bool = False,
) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):
        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [
            box
            for box in eval_boxes[sample_token]
            if box.ego_dist < max_dist[box.__getattribute__("detection_name")]
        ]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [
            box for box in eval_boxes[sample_token] if not box.num_pts == 0
        ]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        sample_anns = nusc.get("sample", sample_token)["anns"]
        bikerack_recs = [
            nusc.get("sample_annotation", ann)
            for ann in sample_anns
            if nusc.get("sample_annotation", ann)["category_name"] == "static_object.bicycle_rack"
        ]
        bikerack_boxes = [
            Box(rec["translation"], rec["size"], Quaternion(rec["rotation"]))
            for rec in bikerack_recs
        ]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.__getattribute__("detection_name") in ["bicycle", "motorcycle"]:
                in_a_bikerack = False
                for bikerack_box in bikerack_boxes:
                    if (
                        np.sum(
                            points_in_box(
                                bikerack_box, np.expand_dims(np.array(box.translation), axis=1)
                            )
                        )
                        > 0
                    ):
                        in_a_bikerack = True
                if not in_a_bikerack:
                    filtered_boxes.append(box)
            else:
                filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes


def get_attr_name(nusc: NuScenes, anno: dict, attr_categories_mapper=lambda x: x) -> str:
    if len(anno["attribute_tokens"]) == 0:
        return "none"
    else:
        attr_names = [nusc.get("attribute", t)["name"] for t in anno["attribute_tokens"]]
        attr_categories = [a.split(".")[0] for a in attr_names]
        if attr_categories_mapper("pedestrian_state") in attr_categories:
            return attr_names[attr_categories.index(attr_categories_mapper("pedestrian_state"))]
        elif attr_categories_mapper("cycle_state") in attr_categories:
            return attr_names[attr_categories.index(attr_categories_mapper("cycle_state"))]
        elif attr_categories_mapper("vehicle_state") in attr_categories:
            return attr_names[attr_categories.index(attr_categories_mapper("vehicle_state"))]
        else:
            raise ValueError(f"invalid attributes: {attr_names}")
