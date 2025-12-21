import json
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import tqdm
from numpy.typing import NDArray
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBox, EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig
from nuscenes.nuscenes import Box
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion


class T4Box(DetectionBox):

    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        # Translation to ego vehicle in meters.
        ego_translation: Tuple[float, float, float] = (0, 0, 0),
        # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
        num_pts: int = -1,
        # The class name used in the detection challenge.
        detection_name: str = "car",
        # GT samples do not have a score.
        detection_score: float = -1.0,
        # Box attribute. Each box can have at most 1 attribute.
        attribute_name: str = "",
    ):
        # Call the grandparents' __init__ method to avoid unnecessary checks in DetectionBox
        EvalBox.__init__(
            self,
            sample_token,
            translation,
            size,
            rotation,
            velocity,
            ego_translation,
            num_pts,
        )
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name


def velocity_clip(velocity: NDArray, max_speed: float = 50.0) -> NDArray:
    """
    Normalize the velocity of the boxes.
    Args:
        velocity (NDArray): Velocity of the boxes.
        boxes (NDArray): Boxes to normalize.
    Returns:
        NDArray: Normalized velocity.
    """
    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        velocity = velocity * (max_speed / speed)
    return velocity


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
            sd_record = nusc.get(
                "sample_data",
                sample_rec["data"]["LIDAR_TOP"],
            )
        elif "LIDAR_CONCAT" in sample_rec["data"]:
            sd_record = nusc.get(
                "sample_data",
                sample_rec["data"]["LIDAR_CONCAT"],
            )
        else:
            raise Exception("Error: LIDAR data not available for sample! Expected either LIDAR_TOP or LIDAR_CONCAT.")
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
            box for box in eval_boxes[sample_token] if box.ego_dist < max_dist[box.__getattribute__("detection_name")]
        ]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        sample_anns = nusc.get("sample", sample_token)["anns"]
        bikerack_recs = [
            nusc.get("sample_annotation", ann)
            for ann in sample_anns
            if nusc.get("sample_annotation", ann)["category_name"] == "static_object.bicycle_rack"
        ]
        bikerack_boxes = [Box(rec["translation"], rec["size"], Quaternion(rec["rotation"])) for rec in bikerack_recs]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.__getattribute__("detection_name") in ["bicycle", "motorcycle"]:
                in_a_bikerack = False
                for bikerack_box in bikerack_boxes:
                    points_in_box_ = points_in_box(
                        bikerack_box,
                        np.expand_dims(
                            np.array(box.translation),
                            axis=1,
                        ),
                    )
                    if np.sum(points_in_box_) > 0:
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


def get_attr_name(
    nusc: NuScenes,
    anno: dict,
) -> List[str]:
    if len(anno["attribute_tokens"]) == 0:
        return []
    else:
        attr_names = [nusc.get("attribute", t)["name"] for t in anno["attribute_tokens"]]
        return attr_names
