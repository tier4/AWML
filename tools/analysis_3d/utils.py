from data_classes import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import numpy.typing as npt
from mmengine.logging import print_log
from t4_devkit import Tier4
from t4_devkit.dataclass import Box3D
from t4_devkit.schema import Sample, EgoPose, CalibratedSensor, SampleData, Scene, Log

from tools.detection3d.t4dataset_converters.t4converter import extract_tier4_data
from tools.detection3d.create_data_t4dataset import get_lidar_token


@dataclass(frozen=True)
class Tier4SampleData:
    """ Data class to save a sample in the Nuscene format. """
    pose_record: EgoPose
    cs_record: CalibratedSensor
    sd_record: SampleData
    scene_record: Scene
    log_record: Log
    boxes: List[Box3D]
    lidar_path: str
    e2g_r_mat: npt.NDArray[np.float64]
    l2e_r_mat: npt.NDArray[np.float64]
    e2g_t: npt.NDArray[np.float64]
    l2e_t: npt.NDArray[np.float64]


def extract_tier4_sample_data(t4: Tier4,
                              sample: Sample) -> Optional[Tier4SampleData]:
    """
    Extract scenario data based on the Tier4 format given a sample record.
    :param t4: Tier4 interface.
    :param sample: A sample record.
    :return: Tier4SampleData.
    """
    lidar_token = get_lidar_token(sample)
    if lidar_token is None:
        print_log(f"sample {sample.token} doesn't have lidar", )
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
    ) = extract_tier4_data(t4, sample, lidar_token)

    return Tier4SampleData(
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
