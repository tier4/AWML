"""Utilities for hdl_utils."""

import inspect
import types

import numpy as np
import torch
from mmdet3d.structures import LiDARInstance3DBoxes
from pyquaternion import Quaternion


class EvalLiDARInstance3DBoxes(LiDARInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates for evaluation.

    Args:
        tensor (torch.Tensor or ndarray or list): Tensor with shape (N, 7/9) containing bbox information.
        score (torch.Tensor or ndarray or list): Tensor with shape (N,) containing scores.
        names (list): List with length N containing class names of objects.

    Attributes:
        score (torch.Tensor): Float vector of N.
        detection_names (list): List with length N.

    """

    def __init__(self, tensor, score, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0), names=None):
        super().__init__(tensor, box_dim, with_yaw, origin)

        assert len(tensor) == len(
            score
        ), "The different size of tensor and score: {len(tensor)} != {len(score)}"

        if isinstance(score, torch.Tensor):
            device = score.device
        else:
            device = torch.device("cpu")
        self._score = torch.as_tensor(score, dtype=torch.float32, device=device)
        self._names = names

    @property
    def score(self) -> torch.Tensor:
        """torch.Tensor: A vector with score of each box."""
        return self._score

    @property
    def names(self) -> [str]:
        """List with length N containing class names of objects."""
        return self._names

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`:
                return a `Boxes` that contains only one box.
            2. `new_boxes = boxes[2:10]`:
                return a slice of boxes.
            3. `new_boxes = boxes[vector]`:
                where vector is a torch.BoolTensor with `length = len(boxes)`.
                Nonzero elements in the vector will be selected.
            Note that the returned Boxes might share storage with this Boxes,
            subject to Pytorch's indexing semantics.

        Returns:
            (EvalLiDARInstance3DBoxes): A new object of EvalLiDARInstance3DBoxes after indexing.

        """
        if isinstance(item, int):
            return EvalLiDARInstance3DBoxes(
                self.tensor[item].view(1, -1),
                box_dim=self.box_dim,
                with_yaw=self.with_yaw,
                score=self.score[item].view(
                    1,
                ),
            )
        b = self.tensor[item]
        s = self.score[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        assert s.dim() == 1, f"Indexing on Boxes with {item} failed to return a matrix!"
        return EvalLiDARInstance3DBoxes(
            b,
            box_dim=self.box_dim,
            with_yaw=self.with_yaw,
            score=s,
        )


class MultiDistanceMatrix(object):
    """An object containing multiple distance matrices."""

    def __init__(self, matrices: dict):
        """Initialize MultiDistanceMatrix.

        Args:
            matrices (dict): a dict with name in keys and matrix in values.

        """
        shape = None
        for matrix in matrices.values():
            assert isinstance(matrix, np.ndarray)
            if shape is None:
                shape = matrix.shape
            else:
                assert matrix.shape == shape

        self.matrices = matrices
        self.shape = shape

    def __len__(self):
        return len(self.matrices.keys())

    def __getitem__(self, key):
        if isinstance(key, str):
            assert key in self.matrices.keys(), f"no such matrix: {key}"
            return self.matrices[key]
        elif isinstance(key, tuple):
            return {name: matrix[key] for name, matrix in self.matrices.items()}
        else:
            raise TypeError(f"unsupported key: {key}")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.matrices[key] = value
        elif isinstance(key, tuple):
            for matrix in self.matrices.values():
                matrix.__setitem__(key, value)
        else:
            raise TypeError(f"unsupported key: {key}")


def _get_attributes(cls):
    """Get class attributes."""
    reserved_attributions = dir(type("dummy", (object,), {})) + ["__annotations__"]
    local_functions = [
        item[0]
        for item in inspect.getmembers(cls)
        if isinstance(item[1], types.MethodType) and item[1].__module__ == cls.__module__
    ]

    return [
        item
        for item in inspect.getmembers(cls)
        if item[0] not in reserved_attributions + local_functions
    ]


def boxes_to_eval_boxes(boxes: list, box_dim=7, origin=(0.5, 0.5, 0)):
    """Convert predicted boxes to EvalLiDARInstance3DBoxes."""

    if len(boxes) == 0:
        return EvalLiDARInstance3DBoxes(
            tensor=torch.zeros((0, box_dim)),
            score=torch.zeros((0,)),
            names=[],
            box_dim=box_dim,
            origin=origin,
        )
    return EvalLiDARInstance3DBoxes(
        tensor=torch.stack([obj["bboxes_3d"] for obj in boxes], dim=0),
        score=[obj["scores_3d"] for obj in boxes],
        names=[obj["detection_names"] for obj in boxes],  # temporary
        box_dim=box_dim,
        origin=origin,
    )


def gt_boxes_to_eval_boxes(gt_boxes: list, box_dim=7):
    """Convert ground truth boxes to EvalLiDARInstance3DBoxes."""

    if box_dim == 9:
        # there is velocity in the gts, so it needs to be concatenated
        bboxes = [
            torch.cat([torch.tensor(gt["bbox_3d"]), torch.tensor(gt["velocity"])])
            for gt in gt_boxes
        ]
    else:
        bboxes = [torch.tensor(gt["bbox_3d"]) for gt in gt_boxes]
    scores = [1.0 for _ in gt_boxes]
    labels = [gt["bbox_label_3d"] for gt in gt_boxes]  # temporary

    return EvalLiDARInstance3DBoxes(
        tensor=torch.stack(bboxes, dim=0),
        score=scores,
        names=labels,
        box_dim=box_dim,
    )
