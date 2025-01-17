# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import numpy as np
from mmdet3d.structures.points import BasePoints
from mmcv.transforms import to_tensor

from mmdet3d.registry import TRANSFORMS,DATASETS
from mmdet3d.structures.points import BasePoints
from autoware_ml.detection3d.datasets.t4dataset import T4Dataset 
from typing import Tuple

import torch
import random
import pickle
import gc



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

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, *args, **kwargs):
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
        sort_items = [x["token"] for x in self.data_list]
        self.datalist = [self.data_list[i] for i in np.argsort(sort_items)]
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
    
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_annot_info(i)
            
            if not self.seq_mode: # for sliding window only
                if prev_scene_token is None:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                elif input_dict['scene_token'] != prev_scene_token:
                    queue.insert(0,queue[0])
                    continue
                else:
                    input_dict.update(dict(prev_exists=True))
            example = self.pipeline(input_dict)

            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[-k-1] is None or ~(queue[-k-1]['gt_labels_3d'] != -1).any()):
                return None
        assert all(x["scene_token"]==queue[0]["scene_token"] for x in queue), "All frames must be from same scene"
        return self.union2one(queue)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_annot_info(index)
        example = self.pipeline(input_dict)
        return example
        
    def union2one(self, queue):
        updated = {}
        for key in self.collect_keys:
            if key != 'img_metas':
                updated[key] = torch.stack([each[key] for each in queue])
            else:
                updated[key] = [each[key] for each in queue]

        for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_bboxes_labels', 'centers_2d', 'depths']:
            if key == 'gt_bboxes_3d':
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

                - sample_idx (str): Sample index.
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
        l2e_matrix = np.eye(4)
        ego_pose =  e2g_matrix @ l2e_matrix # lidar2global
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info.get('sweeps',[]),
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            prev_idx=info.get('prev',None),
            next_idx=info.get('next',None),
            scene_token=info['scene_token'],
            frame_idx=info['sample_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['images'].items():
                img_timestamp.append(cam_info.get('timestamp',-1) / 1e6)
                image_paths.append(cam_info['img_path'])
                intrinsic_mat = np.array(cam_info.get("cam2img",np.eye(3)))
                extrinsic_mat = np.array(cam_info.get("lidar2cam",np.eye(4)))[:3,:]
                intrinsics.append(intrinsic_mat)
                extrinsics.append(extrinsic_mat)
                lidar2img_rts.append(np.concatenate([intrinsic_mat@extrinsic_mat,np.array([[0,0,0,1]])]))
            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or super().get_data_info(index - 1)["scene_token"] != info["scene_token"])
            else:
                prev_exists = None

            input_dict.update(
                dict(
                    images= info['images'],
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                    img_metas=dict(scene_token=info["scene_token"])
                ))
        if not self.test_mode:
            annos = self.parse_ann_info(info)
            # annos.update( 
            #     dict(
            #         bboxes=info['bboxes2d'],
            #         labels=info['labels2d'],
            #         centers2d=info['centers2d'],
            #         depths=info['depths'],
            #         bboxes_ignore=info['bboxes_ignore'])
            # )
            input_dict['ann_info'] = annos
            
        return input_dict


    def prepare_data(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
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

@TRANSFORMS.register_module()
class PETRFormatBundle3D:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, collect_keys, with_gt=True, with_label=True):
        super(PETRFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label
        self.collect_keys = collect_keys
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = results['points'].tensor

        for key in self.collect_keys:
            if key in ['timestamp',  'img_timestamp']:
                 results[key] = to_tensor(np.array(results[key], dtype=np.float64))
            else:
                 results[key] = to_tensor(np.array(results[key], dtype=np.float32))

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = to_tensor(results[key])

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_bboxes_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_bboxes_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_bboxes_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                                                       dtype=np.int64)
        # results = super(PETRFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'collect_keys={self.collect_keys}, with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
