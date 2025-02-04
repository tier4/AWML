import numpy as np

from mmdet3d.registry import DATASETS
from autoware_ml.detection3d.datasets.t4dataset import T4Dataset
from typing import Tuple

import torch
import random
import pickle
import gc
import os


def convert_to_torch(data):
    """
    Recursively iterate through a structure (list, dict, or nested combination),
    and convert all numpy arrays into torch tensors.

    Args:
        data: The input data structure, which could be a list, dict, or any nested structure.

    Returns:
        The modified data structure with numpy arrays converted to torch tensors.
    """
    if isinstance(data, np.ndarray):
        # If the data is a numpy array, convert it to a torch tensor
        return torch.from_numpy(data)
    elif isinstance(data, list):
        # If the data is a list, apply the function recursively to each element
        return [convert_to_torch(item) for item in data]
    elif isinstance(data, dict):
        # If the data is a dictionary, apply the function recursively to each value
        return {key: convert_to_torch(value) for key, value in data.items()}
    elif isinstance(data, tuple):
        # If the data is a tuple, apply the function recursively to each element and return a tuple
        return tuple(convert_to_torch(item) for item in data)
    elif isinstance(data, set):
        # If the data is a set, apply the function recursively to each element and return a set
        return {convert_to_torch(item) for item in data}
    else:
        # If the data is not a recognized structure, return it as-is
        return data


@DATASETS.register_module()
class StreamPETRDataset(T4Dataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(
        self,
        collect_keys,
        seq_mode=False,
        seq_split_num=1,
        num_frame_losses=1,
        queue_length=8,
        random_length=0,
        camera_order=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
        self.camera_order = camera_order

    def _validate_entry(self, info) -> bool:
        """
        Validate the necessary entries in the data info dict
        """
        if not all([x["img_path"] and os.path.exists(x["img_path"]) for x in info["images"].values()]):
            print(
                f"Found frame  {(info['scene_token'],info['token'])} without any image in it, not using it for training"
            )
            return False
        return True

    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize ``self.data_list`` to save memory when launching multiple
        workers in data loading. This function will be called in ``full_init``.

        Hold memory using serialized objects, and data loader workers can use
        shared RAM from master process instead of making a copy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Serialized result and corresponding
            address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        # Serialize data information list avoid making multiple copies of
        # `self.data_list` when iterate `import torch.utils.data.dataloader`
        # with multiple workers.
        sort_items = [(x["scene_token"], x["timestamp"]) for x in self.data_list]
        for i in range(len(self.data_list)):
            self.data_list[i][
                "pre_sample_idx"
            ] = i  # This is necessary to match gts and predictions for frames in testing
        argsorted_indices = sorted(list(range(len(sort_items))), key=lambda i: sort_items[i])
        self.data_list = [self.data_list[i] for i in argsorted_indices if self._validate_entry(self.data_list[i])]
        data_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        # TODO Check if np.concatenate is necessary
        data_bytes = np.concatenate(data_list)
        # Empty cache for preventing making multiple copies of
        # `self.data_info` when loading data multi-processes.
        self.data_list.clear()
        gc.collect()
        return data_bytes, data_address

    def _validate_data(self, queue):
        assert all(x["scene_token"] == queue[0]["scene_token"] for x in queue), "All frames must be from same scene"

    def prepare_temporal_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index - self.queue_length - self.random_length + 1, index))
        if self.random_length:
            random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length :])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_annot_info(i)

            if not self.seq_mode:  # for sliding window only
                if prev_scene_token is None:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict["scene_token"]
                elif input_dict["scene_token"] != prev_scene_token:
                    queue.insert(0, queue[0])
                    continue
                else:
                    input_dict.update(dict(prev_exists=True))
            example = self.pipeline(input_dict)

            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and (queue[-k - 1] is None or ~(queue[-k - 1]["gt_labels_3d"] != -1).any()):
                return None

        self._validate_data(queue)

        return self._union2one(queue)

    def _union2one(self, queue):
        updated = {}
        for key in self.collect_keys:
            if key != "img_metas":
                updated[key] = torch.stack([each[key] for each in queue])
            else:
                updated[key] = [each[key] for each in queue]

        for key in ["gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "gt_bboxes_labels", "centers_2d", "depths"]:
            if key == "gt_bboxes_3d":
                updated[key] = [each[key] for each in queue]
            else:
                updated[key] = [convert_to_torch(each[key]) for each in queue]
        return updated

    def get_annot_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = super().get_data_info(index)
        # standard protocal modified from SECOND.Pytorch
        e2g_matrix = np.array(info["ego2global"])
        l2e_matrix = np.array(info["lidar_points"]["lidar2ego"])
        ego_pose = e2g_matrix @ l2e_matrix  # lidar2global
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        input_dict = dict(
            pts_filename=info["lidar_path"],
            sweeps=info.get("sweeps", []),
            ego_pose=ego_pose,
            ego_pose_inv=ego_pose_inv,
            prev_idx=info.get("prev", None),
            next_idx=info.get("next", None),
            scene_token=info["scene_token"],
            frame_idx=info["token"],
            timestamp=info["timestamp"] / 1e9,
            l2e_matrix=l2e_matrix,
            e2g_matrix=e2g_matrix,
        )

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            
            if self.camera_order:
                camera_order = self.camera_order
            else:
                camera_order = list(info["images"].keys())
                if not self.test_mode:
                    np.random.shuffle(camera_order)

            for cam_type in camera_order :
                cam_info = info["images"][cam_type]
                img_timestamp.append(cam_info["timestamp"] / 1e9)
                image_paths.append(cam_info["img_path"])
                intrinsic_mat = np.array(cam_info["cam2img"])
                extrinsic_mat = np.array(cam_info["lidar2cam"])
                intrinsics.append(intrinsic_mat)
                extrinsics.append(extrinsic_mat)
                lidar2img_rts.append(np.concatenate([intrinsic_mat @ extrinsic_mat[:3, :], np.array([[0, 0, 0, 1]])]))

            prev_exists = not (index == 0 or super().get_data_info(index - 1)["scene_token"] != info["scene_token"])

            input_dict.update(
                dict(
                    images=info["images"],
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                    img_metas=dict(
                        scene_token=info["scene_token"],
                        sample_idx=info["pre_sample_idx"],
                        sample_token=info["token"],
                    ),
                )
            )

        annos = self.parse_ann_info(info)
        input_dict["ann_info"] = annos
        return input_dict

    def prepare_data(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        data = self.prepare_temporal_data(idx)
        return data


def invert_matrix_egopose_numpy(egopose):
    """Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix
