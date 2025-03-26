from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.ticker import MaxNLocator
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
        point_thresholds: List[int],
        sample_ratio: float = 0.3,
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
        self.point_thresholds = point_thresholds
        self.sample_ratio = sample_ratio
        self.full_output_path = self.out_path / self.analysis_dir
        self.full_output_path.mkdir(exist_ok=True, parents=True)

        self.analysis_file_name = "voxel_count_{}.png"
        self.y_axis_label = "Frequency"
        self.x_axis_label = "Number of voxels per frame"
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
        points = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, self.load_dim)
        in_radius = np.logical_and(
            (points[:, 0] >= self.pc_ranges[0]) & (points[:, 0] <= self.pc_ranges[3]),
            (points[:, 1] >= self.pc_ranges[1]) & (points[:, 1] <= self.pc_ranges[4]),
            (points[:, 2] >= self.pc_ranges[2]) & (points[:, 2] <= self.pc_ranges[2]),
        )
        points = points[in_radius]
        return points

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

    def _get_total_voxel_counts(self, points: npt.NDArray[np.float64], point_threshold: int) -> int:
        """ """
        # Normalize the points by dividing by voxel size
        voxel_indices = np.floor(
            (points[:, :3] - np.array([self.pc_ranges[0], self.pc_ranges[1], self.pc_ranges[2]])) / self.voxel_sizes
        ).astype(np.int32)

        # Remove duplicate voxels (points inside the same voxel)
        unique_voxels, unique_counts = np.unique(voxel_indices, axis=0, return_counts=True)

        unique_voxels = unique_voxels[unique_counts >= point_threshold]
        return len(unique_voxels)

    def _compute_scenario_voxel_counts(
        self,
        analysis_data: AnalysisData,
    ) -> Dict[str, List[int]]:
        """Gather voxel counts for each scenario in a dataset."""
        voxel_counts = {i: [] for i in self.point_thresholds}
        for scenario_data in analysis_data.scenario_data.values():
            sample_data = list(scenario_data.sample_data.values())
            selected_sample_data = (
                np.random.choice(sample_data, int(len(sample_data) * self.sample_ratio), replace=False)
                if len(sample_data) > 0
                else sample_data
            )
            for sample in selected_sample_data:
                if sample.lidar_point is None:
                    continue

                points = self._load_points(sample.lidar_point.lidar_path)
                if sample.lidar_sweeps:
                    points = self._load_multisweeps(points, sample.lidar_sweeps)

                for point_threshold in self.point_thresholds:
                    voxel_counts[point_threshold].append(self._get_total_voxel_counts(points, point_threshold))

        return voxel_counts

    def _compute_split_voxel_counts(
        self,
        dataset_analysis_data: Dict[str, AnalysisData],
    ) -> Dict[int, List[int]]:
        """ """
        voxel_counts = {i: [] for i in self.point_thresholds}
        for analysis_data in dataset_analysis_data.values():
            dataset_voxel_counts = self._compute_scenario_voxel_counts(analysis_data)

            for i in self.point_thresholds:
                voxel_counts[i] += dataset_voxel_counts[i]

        return voxel_counts

    def _visualize_voxel_counts(
        self,
        voxel_counts: Dict[int, List[int]],
        split_name: str,
        log_scale: bool = False,
        figsize: tuple[int, int] = (15, 15),
    ) -> None:
        """ """
        columns = len(self.point_thresholds)
        _, axes = plt.subplots(nrows=1, ncols=columns, figsize=figsize)
        percentiles = [0, 25, 50, 75, 95, 100]
        colors = ["blue", "orange", "green", "red", "purple", "brown"]
        # Plot something in each subplot
        for point_threshold, ax in zip(voxel_counts, axes.flatten()):
            voxel_count = voxel_counts[point_threshold]

            p_values = np.percentile(voxel_count, percentiles)
            mean = np.mean(voxel_count)
            std = np.std(voxel_count)
            print_log(
                f"Split name: {split_name}, Point threshold: {point_threshold}, total num of samples: {len(voxel_count)}"
            )

            ax.hist(voxel_count, bins=self.bins, log=log_scale)
            for value, percentile, color in zip(p_values, percentiles, colors):
                ax.axvline(value, color=color, linestyle="dashed", linewidth=2, label=f"P{percentile}:{value:.2f}")

            ax.axvline(mean, color="black", linestyle="dashed", linewidth=2, label=f"mean:{mean:.2f} (std:{std:.2f})")
            ax.set_ylabel(self.y_axis_label)
            ax.set_xlabel(self.x_axis_label)
            ax.set_title(
                f"Voxel counts for {split_name} \n {self.pc_ranges} \n {self.voxel_sizes} \n frames: {len(voxel_count)} \n threshold: {point_threshold}"
            )
            ax.legend(loc=self.legend_loc)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        analysis_file_name = self.full_output_path / self.analysis_file_name.format(split_name)
        plt.savefig(
            fname=analysis_file_name,
            format="png",
            bbox_inches="tight",
        )
        print_log(f"Saved analysis to {analysis_file_name}")
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

            voxel_counts = self._compute_split_voxel_counts(dataset_analysis_data=dataset_voxel_data)
            self._visualize_voxel_counts(
                voxel_counts=voxel_counts, split_name=split_option.value, log_scale=False, figsize=(24, 12)
            )
        print_log(f"Done running {self.__class__.__name__}")
