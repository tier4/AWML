import gc
import math
import os
import pickle
import random
from typing import Tuple

import numpy as np
import torch
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS

from autoware_ml.detection3d.datasets.t4dataset import T4Dataset


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
class StreamPETRNuScenesDataset(T4Dataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(
        self,
        collect_keys,
        seq_mode=True,
        seq_split_num=1,
        num_frame_losses=1,
        queue_length=8,
        random_length=0,
        camera_order=None,
        metainfo={},
        filter_empty_gt=False,
        *args,
        **kwargs,
    ):
        NuScenesDataset.METAINFO = metainfo
        super().__init__(metainfo=metainfo, filter_empty_gt=filter_empty_gt, *args, **kwargs)
        assert seq_mode, "Only supported seq_mode training at the moment"
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
            self._set_group_indices()
        self.camera_order = camera_order

        print(f"Camera corder: {self.camera_order} test_mode: {self.test_mode}")

    def _validate_entry(self, info) -> bool:
        """
        Validate the necessary entries in the data info dict
        """
        if not all([x["img_path"] and os.path.exists(x["img_path"]) for x in info["images"].values()]):
            print(f"Found frame  {(info['sample_idx'])} without any image in it, not using it for training")
            return False
        return True

    def _set_group_indices(self):
        res = []

        curr_sequence = 0
        for idx in range(len(self)):
            if idx != 0 and len(self.get_data_info(idx).get("lidar_sweeps", [])) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)
        flag = np.array(res, dtype=np.int64)
        bin_counts = np.bincount(flag)
        new_flags = []
        curr_new_flag = 0
        for curr_flag in range(len(bin_counts)):
            curr_sequence_length = np.array(
                list(range(0, bin_counts[curr_flag], math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                + [bin_counts[curr_flag]]
            )

            for sub_seq_idx in curr_sequence_length[1:] - curr_sequence_length[:-1]:
                for _ in range(sub_seq_idx):
                    new_flags.append(curr_new_flag)
                curr_new_flag += 1

        assert len(new_flags) == len(flag)
        assert len(np.bincount(new_flags)) == len(np.bincount(flag)) * self.seq_split_num
        self.flag = np.array(new_flags, dtype=np.int64)

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
        sort_items = [x["sample_idx"] for x in self.data_list]
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
        input_dict = self.get_annot_info(index)
        if self.seq_mode:
            input_dict.update(dict(prev_exists=self.flag[index - 1] == self.flag[index]))
        else:
            raise NotImplementedError("Sliding window is not implemented for nuscenes")
        example = self.pipeline(input_dict)

        queue = [example]

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

        for key in [
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_bboxes",
            "gt_bboxes_labels",
            "centers_2d",
            "depths",
        ]:  # , "gt_bboxes_gen", "gt_bboxes_labels_gen", "centers_2d_gen", "depths_gen"]:
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
            sweeps=info.get("lidar_sweeps", []),
            ego_pose=ego_pose,
            ego_pose_inv=ego_pose_inv,
            prev_idx=info.get("prev", None),
            next_idx=info.get("next", None),
            scene_token=None,
            frame_idx=info["token"],
            timestamp=info["timestamp"],
            l2e_matrix=l2e_matrix,
            e2g_matrix=e2g_matrix,
        )

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []

            gt_bboxes = []
            centers_2d = []
            gt_bboxes_labels = []
            depths = []

            if self.camera_order:
                camera_order = self.camera_order
            else:
                camera_order = list(info["images"].keys())
                if not self.test_mode:
                    random.shuffle(camera_order)

            info["images"] = {x: info["images"][x] for x in camera_order}

            for cam_type in info["images"]:
                cam_info = info["images"][cam_type]
                img_timestamp.append(cam_info["timestamp"])
                image_paths.append(cam_info["img_path"])
                intrinsic_mat = np.array(cam_info["cam2img"])
                extrinsic_mat = np.array(cam_info["lidar2cam"])  # @ np.linalg.inv(l2e_matrix)
                intrinsics.append(intrinsic_mat)
                extrinsics.append(extrinsic_mat)

                gt_bboxes.append(np.array([x["bbox"] for x in info["cam_instances"][cam_type]]))
                centers_2d.append(np.array([x["center_2d"] for x in info["cam_instances"][cam_type]]))
                gt_bboxes_labels.append(np.array([x["bbox_label"] for x in info["cam_instances"][cam_type]]))
                depths.append(np.array([x["depth"] for x in info["cam_instances"][cam_type]]))

                lidar2img_rts.append(np.concatenate([intrinsic_mat @ extrinsic_mat[:3, :], np.array([[0, 0, 0, 1]])]))

            input_dict.update(
                dict(
                    images=info["images"],
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    img_metas=dict(
                        scene_token=None,
                        sample_idx=info["pre_sample_idx"],
                        order_index=info["pre_sample_idx"],
                        sample_token=info["token"],
                        filenames=image_paths,
                    ),
                    gt_bboxes=gt_bboxes,
                    centers_2d=centers_2d,
                    gt_bboxes_labels=gt_bboxes_labels,
                    depths=depths,
                    # cam_instances=info["cam_instances"],
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
