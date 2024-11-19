from data_classes import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import numpy.typing as npt
from mmengine.logging import print_log
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

from tools.detection3d.t4dataset_converters.t4converter import extract_nuscenes_data
from tools.detection3d.create_data_t4dataset import get_lidar_token


@dataclass(frozen=True)
class NuSceneSampleData:
    """ Data class to save a sample in the Nuscene format. """
    pose_record: Dict[str, Any]
    cs_record: Dict[str, Any]
    sd_record: Dict[str, Any]
    scene_record: Dict[str, Any]
    log_record: Dict[str, Any]
    boxes: List[Box]
    lidar_path: str
    e2g_r_mat: npt.NDArray[np.float64]
    l2e_r_mat: npt.NDArray[np.float64]
    e2g_t: npt.NDArray[np.float64]
    l2e_t: npt.NDArray[np.float64]


def extract_nuscene_sample_data(
        nusc: NuScenes, sample: Dict[str, Any]) -> Optional[NuSceneSampleData]:
    """
    Extract scenario data based on the NuScene format given a sample record.
    :param nusc: Nuscene interface.
    :param sample: A sample record.
    :return: NusceneSampleData.
    """
    lidar_token = get_lidar_token(sample)
    if lidar_token is None:
        print_log(f"sample {sample['token']} doesn't have lidar", )
        return

    (
        pose_record,
        cs_record,
        sd_record,
        scene_record,
        log_record,
        boxes,
        lidar_path,
        e2g_r_mat,
        l2e_r_mat,
        e2g_t,
        l2e_t,
    ) = extract_nuscenes_data(nusc, sample, lidar_token)

    return NuSceneSampleData(
        pose_record=pose_record,
        cs_record=cs_record,
        sd_record=sd_record,
        scene_record=scene_record,
        log_record=log_record,
        boxes=boxes,
        lidar_path=lidar_path,
        e2g_r_mat=e2g_r_mat,
        l2e_r_mat=l2e_r_mat,
        e2g_t=e2g_t,
        l2e_t=l2e_t)


def get_box_attrs(nusc: NuScenes, boxes: List[Box]) -> Dict[str, List[str]]:
    """
    Get attributes for every box from Nuscenes.
    :param nusc: Nuscene interface.
    :param boxes: A list of Box.
    :return: A dict of {box token: list of attributes}.
    """
    box_attrs = {}
    for box in boxes:
        sample_annotation = nusc.get("sample_annotation", box.token)
        attribute_tokens = sample_annotation['attribute_tokens']
        attr_names = [
            nusc.get("attribute", attribute_token)["name"]
            for attribute_token in attribute_tokens
        ]
        box_attrs[box.token] = attr_names

    return box_attrs
