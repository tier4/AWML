import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.ticker import MaxNLocator
from mmengine.logging import print_log
from t4_devkit.schema.tables.sample import Sample

from tools.analysis_3d.callbacks.callback_interface import AnalysisCallbackInterface
from tools.analysis_3d.data_classes import (
    AnalysisData,
    DatasetSplitName,
    Detection3DBox,
    LidarPoint,
    LidarSweep,
    ScenarioData,
)
from tools.analysis_3d.split_options import SplitOptions


@dataclass(frozen=True)
class CategoryPercentiles:
    """Class to save percentiles of a category."""

    category_name: str
    percentiles: Dict[str, float]
    mean: float
    std: float


@dataclass(frozen=True)
class BBoxPair:

    instance_name: str
    instance_token: str
    sample_token: str
    category_name: str
    displacement_x: float
    displacement_y: float
    displacement_z: float
    distance: float
    timestamp_diff: float
    yaw_diff: float
    velocity_diff: float
    angle: float  # in degree
    timestamp_index: int


class SeuquenceBBoxDiffAnalysisCallback(AnalysisCallbackInterface):
    """Compute diffs of translations for every consecutive frame."""

    def __init__(
        self,
        data_root_path: Path,
        out_path: Path,
        bins: int = 100,
        remapping_classes: Optional[Dict[str, str]] = None,
        analysis_dir: str = "translation_diffs",
    ) -> None:
        """
        :param out_path: Path to save outputs.
        :param analysis_dir: Folder name to save outputs.
        :param remapping_classes: Set if compute frequency of every category after remapping.
        """
        super(AnalysisCallbackInterface, self).__init__()
        self.data_root_path = data_root_path
        self.out_path = out_path
        self.analysis_dir = analysis_dir
        self.full_output_path = self.out_path / self.analysis_dir
        self.full_output_path.mkdir(exist_ok=True, parents=True)

        self.analysis_file_name = "distance_{}_{}.png"
        self.analysis_bin_file_name = "distance_hist_{}_{}.png"
        self.y_axis_label = "Frequency"
        self.x_axis_label = "Difference between two frames"
        self.legend_loc = "upper right"
        self.bins = bins
        self.remapping_classes = remapping_classes
        self.weights = {"car": 8.0, "pedestrian": 5.0}
        self.default_weight = 3.0

    def _write_abnormal_instances_to_csv(self, file_name: str, columns: List[str], data: List[Any]) -> None:
        """Write abnormal instances to CSV."""
        csv_file_name = self.full_output_path / file_name
        with open(csv_file_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            for row in data:
                writer.writerow(row)
        print_log(f"Saved data plot to {csv_file_name}")

    def _get_abnormal_annotations(
        self,
        dataset_bbox_pairs: Dict[str, Dict[str, Dict[str, BBoxPair]]],
        category_percentiles: Dict[str, Dict[str, CategoryPercentiles]],
        attribute_name: str,
    ) -> Tuple[List[str], List[List[Any]]]:
        """ """
        columns = ["t4dataset", "instance_token", "instance_name"] + [f"frame_{i+1}" for i in range(30)]
        data = []
        # Gather translation differences for each instance
        for scene_token, scene_data in dataset_bbox_pairs.items():
            for instance_name, instance_data in scene_data.items():
                frames = [False] * 30
                # Extract the category name from the instance name
                category_name, instance_token, name = instance_name.split("/")
                category_perceptile = category_percentiles[category_name][attribute_name]
                instance_row = [scene_token, instance_token, name]
                weight = self.weights.get(category_name, self.default_weight)
                for sample_token, bbox_pair in instance_data.items():
                    q3 = category_perceptile.percentiles["Q3"]
                    q1 = category_perceptile.percentiles["Q1"]
                    iqr = q3 - q1
                    dist_threshold = q3 + iqr * weight
                    value = bbox_pair.__getattribute__(attribute_name)
                    if value > dist_threshold:
                        frames[bbox_pair.timestamp_index] = True
                        frames[bbox_pair.timestamp_index + 1] = True

                if any(frames):
                    instance_row += frames
                    data.append(instance_row)

        return columns, data

    def get_category_bbox_pairs(
        self, dataset_bbox_pairs: Dict[str, Dict[str, Dict[str, BBoxPair]]]
    ) -> Dict[str, List[tuple]]:
        """
        :param dataset_bbox_pairs: {scene: {instance_name: {sample_token: BboxPair}}}.
        :return: {category_name: [translation_diff]}.
        """
        category_bbox_pairs = defaultdict(list)
        # Gather translation differences for each instance
        for scene_token, scene_data in dataset_bbox_pairs.items():
            for instance_name, sample_data in scene_data.items():
                for sample_token, bbox_pair in sample_data.items():
                    category_bbox_pairs[bbox_pair.category_name].append(bbox_pair)

        return category_bbox_pairs

    def plot_dataset_distance_boxplot(
        self,
        dataset_name: str,
        category_bbox_pairs: Dict[str, List[BBoxPair]],
        figsize: tuple[int, int] = (16, 16),
    ) -> Dict[str, Dict[str, CategoryPercentiles]]:
        """
        :param category_translation_diffs: {category_name: [BBoxPair]}.
        """
        ax_names = ["X", "Y", "Z", "Distance"]
        attribute_names = ["displacement_x", "displacement_y", "displacement_z", "distance"]
        category_percentiles = defaultdict(lambda: defaultdict(CategoryPercentiles))
        for category_name, bbox_pairs in category_bbox_pairs.items():
            # Plot translation differences for each category and differences in translations
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize)
            axes = axes.flatten()
            for index in range(4):
                attribute_name = attribute_names[index]
                values = [abs(bbox_pair.__getattribute__(attribute_name)) for bbox_pair in bbox_pairs]
                ax = axes[index]
                ax_name = ax_names[index]
                ax.boxplot(values, vert=True, patch_artist=True)

                # Compute quartiles and IQR
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                median = np.percentile(values, 50)
                iqr = q3 - q1
                percentiles = {
                    "Q1": q1,
                    "Q3": q3,
                    "Median": median,
                }
                mean = np.mean(values)
                std = np.std(values)
                category_percentiles[category_name] = {
                    attribute_name: CategoryPercentiles(
                        category_name=category_name,
                        percentiles=percentiles,
                        mean=mean,
                        std=std,
                    )
                }

                # Annotate Q1, Q3, and IQR
                ax.annotate(
                    f"Q1 = {q1:.2f}",
                    xy=(1.1, q1),
                    xytext=(1.2, q1),
                    arrowprops=dict(facecolor="blue", shrink=0.05),
                    fontsize=10,
                )
                ax.annotate(
                    f"Q3 = {q3:.2f}",
                    xy=(1.1, q3),
                    xytext=(1.2, q3),
                    arrowprops=dict(facecolor="green", shrink=0.05),
                    fontsize=10,
                )
                ax.annotate(
                    f"Median = {median:.2f}",
                    xy=(1.1, median),
                    xytext=(1.2, median),
                    arrowprops=dict(facecolor="orange", shrink=0.05),
                    fontsize=10,
                )
                ax.text(
                    0.75, (q1 + q3) / 2, f"IQR = {iqr:.2f}", fontsize=12, color="purple", verticalalignment="center"
                )
                ax.axhline(
                    mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean:.2f}, std = {std:.2f}"
                )
                ax.set_ylabel("Distance")
                ax.set_xticks([1])
                ax.set_xticklabels([ax_name])
                ax.legend()

            # Save the plot
            plot_file_name = self.full_output_path / self.analysis_file_name.format(category_name, dataset_name)
            fig.suptitle(category_name)
            plt.tight_layout()
            plt.savefig(plot_file_name)
            print_log(f"Saved displacement plot to {plot_file_name}")
            plt.close()
        return category_percentiles

    def plot_dataset_distance_hist(
        self,
        dataset_name: str,
        category_bbox_pairs: Dict[str, List[BBoxPair]],
        figsize: tuple[int, int] = (16, 16),
    ) -> Dict[str, CategoryPercentiles]:
        """
        :param category_translation_diffs: {category_name: [translation_diff]}.
        """
        percentiles = [0, 25, 50, 75, 95, 98, 99, 100]
        colors = ["blue", "orange", "green", "red", "purple", "brown", "olive", "pink"]
        ax_names = ["X", "Y", "Z", "Distance"]
        attribute_names = ["displacement_x", "displacement_y", "displacement_z", "distance"]
        category_percentiles = defaultdict(lambda: defaultdict(CategoryPercentiles))
        means = defaultdict(list)
        for category_name, bbox_pairs in category_bbox_pairs.items():
            # Plot translation differences for each category and differences in translations
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize)
            axes = axes.flatten()
            for index in range(4):
                attribute_name = attribute_names[index]
                values = [abs(bbox_pair.__getattribute__(attribute_name)) for bbox_pair in bbox_pairs]
                ax = axes[index]
                ax_name = ax_names[index]

                p_values = np.percentile(values, percentiles)
                mean = np.mean(values)
                std = np.std(values)

                category_percentiles[category_name] = {
                    attribute_name: CategoryPercentiles(
                        category_name=category_name,
                        percentiles={f"P{percentile}": p_value for percentile, p_value in zip(percentiles, p_values)},
                        mean=mean,
                        std=std,
                    )
                }

                ax.hist(values, bins=self.bins, log=True)
                for value, percentile, color in zip(p_values, percentiles, colors):
                    ax.axvline(value, color=color, linestyle="dashed", linewidth=2, label=f"P{percentile}:{value:.2f}")

                ax.axvline(
                    mean, color="black", linestyle="dashed", linewidth=2, label=f"mean:{mean:.2f} (std:{std:.2f})"
                )
                ax.set_ylabel("Frequency")
                ax.set_xlabel("Distance")
                ax.set_title(ax_name)
                ax.legend(loc=self.legend_loc)

            # Save the plot
            plot_file_name = self.full_output_path / self.analysis_bin_file_name.format(category_name, dataset_name)
            fig.suptitle(category_name)
            plt.tight_layout()
            plt.savefig(plot_file_name)
            print_log(f"Saved translation diff plot to {plot_file_name}")
            plt.close()
        return means

    def _get_bbox_pair(
        self,
        sample_token: str,
        current_bbox: Detection3DBox,
        next_bbox: Detection3DBox,
        current_timstamp: int,
        next_timestamp: int,
        timestamp_index: int,
    ) -> BBoxPair:
        """ """
        # Compute translation difference in x, y, z
        current_bbox_position = current_bbox.box.position
        next_bbox_position = next_bbox.box.position

        displacement = current_bbox_position - next_bbox_position
        dist = np.linalg.norm(current_bbox_position - next_bbox_position)

        dot = np.dot(current_bbox_position, next_bbox_position)
        norms = np.linalg.norm(current_bbox_position) * np.linalg.norm(next_bbox_position)

        angle_rad = np.arccos(np.clip(dot / norms, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        # Convert to seconds
        timestamp_diff = (abs(current_timstamp - next_timestamp)) / 1e6

        velocity_diff = (
            current_bbox.box.velocity - next_bbox.box.velocity
            if current_bbox.box.velocity is not None or next_bbox.box.velocity
            else np.nan
        )

        if self.remapping_classes is not None:
            category_name = self.remapping_classes.get(
                current_bbox.box.semantic_label.name, current_bbox.box.semantic_label.name
            )
        else:
            category_name - current_bbox.box.semantic_label.name

        yaw_diff = current_bbox.box.rotation.yaw_pitch_roll[0] - next_bbox.box.rotation.yaw_pitch_roll[0]
        return BBoxPair(
            instance_name=current_bbox.instance_name,
            instance_token=current_bbox.box.uuid,
            category_name=category_name,
            sample_token=sample_token,
            displacement_x=displacement[0],
            displacement_y=displacement[1],
            displacement_z=displacement[2],
            distance=dist,
            angle=angle_deg,
            velocity_diff=velocity_diff,
            yaw_diff=yaw_diff,
            timestamp_diff=timestamp_diff,
            timestamp_index=timestamp_index,
        )

    def compute_scenario_bbox_pairs(self, scenario_data: ScenarioData) -> Dict[str, Dict[str, tuple]]:
        """Compute difference of pair of bbox between two frames."""
        # instance_name: sample_token: []
        bbox_pairs: Dict[str, Dict[str, tuple]] = defaultdict(lambda: defaultdict(BBoxPair))

        sample_data = sorted(scenario_data.sample_data.values(), key=lambda x: x.timestamp)

        # Convert box data to {sample: instance_token: box}
        sample_instance_box: Dict[str, Dict[str, Detection3DBox]] = {
            sample.sample_token: {
                detection_3d_box.box.uuid: detection_3d_box for detection_3d_box in sample.detection_3d_boxes
            }
            for sample in sample_data
        }

        # {sample_token: timestamp}
        sample_timestamps: Dict[str, int] = {sample.sample_token: sample.timestamp for sample in sample_data}

        for index, sample in enumerate(sample_data):
            sample_token = sample.sample_token
            if sample.next_sample_token is None:
                continue

            for detection_3d_box in sample.detection_3d_boxes:
                box_category_name = detection_3d_box.box.semantic_label.name
                if self.remapping_classes is not None:
                    # If no category found from the remapping, then it uses the original category name
                    box_category_name = self.remapping_classes.get(box_category_name, box_category_name)

                # Get the next instance
                next_sample: Dict[str, Detection3DBox] = sample_instance_box.get(sample.next_sample_token, None)
                if next_sample is None:
                    continue

                next_instance_box: Detection3DBox = next_sample.get(detection_3d_box.box.uuid, None)
                if next_instance_box is None:
                    continue

                instance_name = f"{box_category_name}/{detection_3d_box.box.uuid}/{detection_3d_box.instance_name}"
                next_sample_timestamp = sample_timestamps.get(sample.next_sample_token, np.nan)
                bbox_pairs[instance_name][sample_token] = self._get_bbox_pair(
                    sample_token=sample_token,
                    current_bbox=detection_3d_box,
                    next_bbox=next_instance_box,
                    current_timstamp=sample.timestamp,
                    next_timestamp=next_sample_timestamp,
                    timestamp_index=index,
                )

        return bbox_pairs

    def run(self, dataset_split_analysis_data: Dict[DatasetSplitName, AnalysisData]) -> None:
        """Inherited, check the superclass."""
        print_log(f"Running {self.__class__.__name__}")
        # Convert to {dataset: AnalysisData}
        dataset_analysis_data: Dict[str, List[AnalysisData]] = defaultdict(list)
        for dataset_split_name, analysis_data in dataset_split_analysis_data.items():
            dataset_analysis_data[dataset_split_name.dataset_version].append(analysis_data)

        for dataset_name, analysis_data in dataset_analysis_data.items():
            scene_bbox_pairs = defaultdict(lambda: defaultdict(lambda: defaultdict(BBoxPair)))
            for analysis in analysis_data:
                # scene: sample: instance
                for scene_token, scenario_data in analysis.scenario_data.items():
                    bbox_pairs = self.compute_scenario_bbox_pairs(
                        scenario_data=scenario_data,
                    )
                    scene_bbox_pairs[scene_token] = bbox_pairs

            category_bbox_pairs = self.get_category_bbox_pairs(scene_bbox_pairs)
            dataset_version = dataset_split_name.dataset_version
            category_percentiles = self.plot_dataset_distance_boxplot(
                dataset_name=dataset_version, category_bbox_pairs=category_bbox_pairs
            )

            # Write abnormal instances
            column, data = self._get_abnormal_annotations(
                dataset_bbox_pairs=scene_bbox_pairs,
                category_percentiles=category_percentiles,
                attribute_name="distance",
            )
            self._write_abnormal_instances_to_csv(
                file_name=f"iqr_abnormal_distance_instances_{dataset_version}.csv",
                columns=column,
                data=data,
            )
            _ = self.plot_dataset_distance_hist(dataset_name=dataset_version, category_bbox_pairs=category_bbox_pairs)
        print_log(f"Done running {self.__class__.__name__}")
