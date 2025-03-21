from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from mmengine.logging import print_log

from tools.analysis_3d.callbacks.callback_interface import AnalysisCallbackInterface
from tools.analysis_3d.data_classes import AnalysisData, DatasetSplitName, LidarPoint, LidarSweep
from tools.analysis_3d.split_options import SplitOptions


class VoxelNumAnalysisCallback(AnalysisCallbackInterface):
    """Compute number of voxels for every pointcloud and its multiple sweeps."""

    def __init__(
        self,
        data_root_path: Path,
        out_path: Path,
        pc_ranges: List[float],
        voxel_sizes: List[float],
        load_dim: int = 5,
        use_dim: List[int] = [0, 1, 2],
        sweeps_num: int = 1,
        remove_close: bool = True,
        analysis_dir: str = "voxel_nums",
        bins: int = 100,
    ) -> None:
        """
        :param out_path: Path to save outputs.
        :param analysis_dir: Folder name to save outputs.
        :param remapping_classes: Set if compute frequency of every category after remapping.
        """
        super(AnalysisCallbackInterface, self).__init__()
        self.data_root_path = data_root_path
        self.out_path = out_path
        self.pc_ranges = pc_ranges
        self.voxel_sizes = voxel_sizes
        self.analysis_dir = analysis_dir
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.remove_close = remove_close
        self.full_output_path = self.out_path / self.analysis_dir
        self.full_output_path.mkdir(exist_ok=True, parents=True)

        self.analysis_file_name = "voxel_count_{}_{}_{}.png"
        self.y_axis_label = "Number of voxels"
        self.x_axis_label = "Bins"
        self.legend_loc = "upper right"
        self.bins = bins

    def _remove_close(self, points: npt.NDArray[np.float32], radius: float = 1.0) -> npt.NDArray[np.float32]:
        """Remove point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray | :obj:`BasePoints`: Points after removing.
        """
        x_filt = np.abs(points[:, 0]) < radius
        y_filt = np.abs(points[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def _load_points(self, pcd_file: str) -> npt.NDArray[np.float32]:
        """ """
        pcd_file = self.data_root_path / pcd_file
        return np.fromfile(pcd_file, dtype=np.float32).reshape(-1, self.load_dim)

    def _load_multisweeps(self, points: npt.NDArray[np.float32], sweeps: List[LidarSweep]) -> npt.NDArray[np.float32]:
        """ """
        points = points[:, self.use_dim]
        sweep_points_list = [points]

        choices = np.random.choice(len(sweeps), self.sweeps_num, replace=False)

        for idx in choices:
            sweep: LidarSweep = sweeps[idx]
            points_sweep = self._load_points(sweep.lidar_path)
            if self.remove_close:
                points_sweep = self._remove_close(points_sweep)
                points_sweep = points_sweep[:, self.use_dim]
                sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        return points

    def _get_total_voxel_counts(self, points: npt.NDArray[np.float64]) -> int:
        """ """
        # Normalize the points by dividing by voxel size
        voxel_indices = np.floor(
            (points - np.array([self.pc_ranges[0], self.pc_ranges[1], self.pc_ranges[2]])) / self.voxel_sizes
        ).astype(np.int32)

        # Remove duplicate voxels (points inside the same voxel)
        unique_voxels = np.unique(voxel_indices, axis=0)

        return len(unique_voxels)

    def _visualize_dataset_voxel_counts(
        self,
        dataset_analysis_data: Dict[str, AnalysisData],
        split_name: str,
        log_scale: bool = False,
        figsize: tuple[int, int] = (15, 15),
    ) -> None:
        """ """
        for dataset_name, analysis_data in dataset_analysis_data.items():
            self._visualize_voxel_counts(analysis_data, split_name, log_scale, figsize)

    def _visualize_voxel_counts(
        self,
        analysis_data: AnalysisData,
        split_name: str,
        log_scale: bool = False,
        figsize: tuple[int, int] = (15, 15),
    ) -> None:
        """ """
        voxel_counts = []
        for scenario_data in analysis_data.scenario_data.values():
            for sample_data in scenario_data.sample_data.values():
                if sample_data.lidar_point is None:
                    continue

                points = self._load_points(sample_data.lidar_point.lidar_path)
                if sample_data.lidar_sweeps:
                    points = self._load_multisweeps(points, sample_data.lidar_sweeps)

            voxel_counts.append(self._get_total_voxel_counts(points))

        _, ax = plt.subplots(figsize=figsize)
        print_log(f"Total num of voxels: {len(voxel_counts)}")
        ax.hist(voxel_counts, bins=self.bins, log=log_scale)
        ax.set_ylabel(self.y_axis_label)
        ax.set_xlabel(self.x_axis_label)
        ax.set_title(f"Voxel counts for {split_name} \n {self.pc_ranges} \n {self.voxel_sizes}")
        plt.tight_layout()
        analysis_file_name = self.full_output_path / self.analysis_file_name.format(
            split_name, self.pc_ranges, self.voxel_sizes
        )
        plt.savefig(
            fname=analysis_file_name,
            format="png",
            bbox_inches="tight",
        )
        plt.close()

    def run(self, dataset_split_analysis_data: Dict[DatasetSplitName, AnalysisData]) -> None:
        """Inherited, check the superclass."""
        print_log(f"Running {self.__class__.__name__}")
        for split_option in SplitOptions:
            dataset_voxel_data = {}
            for dataset_split_name, analysis_data in dataset_split_analysis_data.items():
                split_name = dataset_split_name.split_name
                if split_name != split_option.value:
                    continue
                dataset_voxel_data[dataset_split_name.dataset_version] = analysis_data

            self._visualize_dataset_voxel_counts(dataset_analysis_data=dataset_voxel_data, split_name=split_name)
        print_log(f"Done running {self.__class__.__name__}")
