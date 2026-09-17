"""Point-cloud loaders that can read frames from per-scene HDF5 files.

A ``lidar_path`` of the form ``.../data/LIDAR_CONCAT_<profile>.h5/<frame_id>``
is read as ``/frames/<frame_id>/points`` from that file (layout produced by
``hdf5/lidar_concat_to_hdf5.py``). Any other path falls back to the stock
mmdet3d behaviour, so one pipeline serves both ``.pcd.bin`` and HDF5 infos.
"""

import os
from typing import Dict, Optional, Tuple

import h5py
import hdf5plugin  # noqa: F401  # registers the Blosc2 filter in every (forked) worker
import numpy as np
from mmdet3d.datasets.transforms import LoadPointsFromFile, LoadPointsFromMultiSweeps
from mmdet3d.registry import TRANSFORMS

_HDF5_SEPARATOR = ".h5/"


def split_hdf5_path(path: str) -> Optional[Tuple[str, str]]:
    """Return ``(h5_file, frame_id)`` for a virtual HDF5 frame path, else ``None``."""
    head, separator, frame_id = path.rpartition(_HDF5_SEPARATOR)
    if not separator:
        return None
    return head + ".h5", frame_id


class _HDF5PointsMixin:
    """Override ``_load_points`` with an HDF5 reader backed by per-process handles.

    Dataloader workers are forked from a process that may already hold handles;
    those are unusable in the child, so the cache is keyed by pid and dropped
    (without closing) whenever the pid changes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._h5_files: Dict[str, h5py.File] = {}
        self._h5_pid: Optional[int] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_files"] = {}
        state["_h5_pid"] = None
        return state

    def _h5_file(self, path: str) -> h5py.File:
        pid = os.getpid()
        if pid != self._h5_pid:
            self._h5_files = {}
            self._h5_pid = pid
        handle = self._h5_files.get(path)
        if handle is None:
            handle = self._h5_files[path] = h5py.File(path, "r")
        return handle

    def _load_points(self, pts_filename: str) -> np.ndarray:
        parts = split_hdf5_path(pts_filename)
        if parts is None:
            return super()._load_points(pts_filename)
        h5_path, frame_id = parts
        points = self._h5_file(h5_path)["frames"][frame_id]["points"][()]
        # Callers reshape to (-1, load_dim); the array is already writable.
        return points.reshape(-1)


@TRANSFORMS.register_module()
class LoadPointsFromFileHDF5(_HDF5PointsMixin, LoadPointsFromFile):
    """``LoadPointsFromFile`` that also accepts ``<file>.h5/<frame_id>`` paths."""


@TRANSFORMS.register_module()
class LoadPointsFromMultiSweepsHDF5(_HDF5PointsMixin, LoadPointsFromMultiSweeps):
    """``LoadPointsFromMultiSweeps`` that also accepts ``<file>.h5/<frame_id>`` paths."""
