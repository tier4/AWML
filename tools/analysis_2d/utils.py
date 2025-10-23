from typing import List, Optional

import mmengine
import numpy as np
import numpy.typing as npt
from data_classes import dataclass
from nptyping import NDArray
from t4_devkit import Tier4
from t4_devkit.dataclass import Box2D
from t4_devkit.schema import CalibratedSensor, EgoPose, Log, Sample, SampleData, Scene
from t4_devkit.typing import CameraIntrinsicLike


@dataclass(frozen=True)
class Tier4SampleData:
    """Data class to save a sample in the Nuscene format."""

    pose_record: EgoPose
    cs_record: CalibratedSensor
    sd_record: SampleData
    scene_record: Scene
    log_record: Log
    boxes: List[Box2D]
    camera_path: str
    e2g_r_mat: npt.NDArray[np.float64]
    l2e_r_mat: npt.NDArray[np.float64]
    e2g_t: npt.NDArray[np.float64]
    l2e_t: npt.NDArray[np.float64]
    camera_intrinsics: CameraIntrinsicLike


def get_camera_token(sample_rec: Sample) -> Optional[str]:
    data_dict = sample_rec.data
    for key in data_dict.keys():
        if "CAM" in key:
            return data_dict[key]
    return None


def extract_tier4_data(t4: Tier4, sample: Sample) -> tuple[
    EgoPose,
    CalibratedSensor,
    SampleData,
    Scene,
    Log,
    list[Box2D],
    str,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    CameraIntrinsicLike,
]:
    """
    Extract scenario data based on the Tier4 format given a sample record.
    :param t4: Tier4 interface.
    :param sample: A sample record.
    :return: Tier4SampleData.
    """
    camera_token = get_camera_token(sample)
    if camera_token is None:
        mmengine.print_log(
            f"sample {sample.token} doesn't have camera",
        )
        return

    sd_record: SampleData = t4.get("sample_data", camera_token)
    cs_record: CalibratedSensor = t4.get("calibrated_sensor", sd_record.calibrated_sensor_token)
    pose_record: EgoPose = t4.get("ego_pose", sd_record.ego_pose_token)

    camera_path, boxes, camera_intrinsics = t4.get_sample_data(camera_token, as_3d=False)
    mmengine.check_file_exist(camera_path)

    scene_record: Scene = t4.get("scene", sample.scene_token)
    log_record = t4.get("log", scene_record.log_token)

    l2e_t = cs_record.translation
    e2g_t = pose_record.translation
    l2e_r = cs_record.rotation
    e2g_r = pose_record.rotation
    l2e_r_mat = l2e_r.rotation_matrix
    e2g_r_mat = e2g_r.rotation_matrix
    return (
        pose_record,
        cs_record,
        sd_record,
        scene_record,
        log_record,
        boxes,
        camera_path,
        e2g_r_mat,
        l2e_r_mat,
        e2g_t,
        l2e_t,
        camera_intrinsics,
    )


def extract_tier4_sample_data(t4: Tier4, sample: Sample) -> Optional[Tier4SampleData]:
    """
    Extract scenario data based on the Tier4 format given a sample record.
    :param t4: Tier4 interface.
    :param sample: A sample record.
    :return: Tier4SampleData.
    """

    (
        pose_record,
        cs_record,
        sd_record,
        scene_record,
        log_record,
        boxes,
        camera_path,
        e2g_r_mat,
        l2e_r_mat,
        e2g_t,
        l2e_t,
        camera_intrinsics,
    ) = extract_tier4_data(t4, sample)

    return Tier4SampleData(
        pose_record=pose_record,
        cs_record=cs_record,
        sd_record=sd_record,
        scene_record=scene_record,
        log_record=log_record,
        boxes=boxes,
        camera_path=camera_path,
        e2g_r_mat=e2g_r_mat,
        l2e_r_mat=l2e_r_mat,
        e2g_t=e2g_t,
        l2e_t=l2e_t,
        camera_intrinsics=camera_intrinsics,
    )
