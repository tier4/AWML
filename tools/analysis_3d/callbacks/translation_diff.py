import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.ticker import MaxNLocator
from mmengine.logging import print_log

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


class TranslationDiffAnalysisCallback(AnalysisCallbackInterface):
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

        self.analysis_file_name = "translation_diff_{}_{}.png"
        self.analysis_bin_file_name = "translation_diff_bin_{}_{}.png"
        self.y_axis_label = "Frequency"
        self.x_axis_label = "Difference between two frames"
        self.legend_loc = "upper right"
        self.bins = bins
        self.remapping_classes = remapping_classes

    def compute_mapping_sample_to_frame_index(self, scenario_data: ScenarioData) -> Dict[str, Dict[str, tuple]]:
        """Compute translation difference between two frames."""
        # {sample_token: index}}
        sample_data = sorted(scenario_data.sample_data.values(), key=lambda x: x.timestamp)
        return {sample.sample_token: index for sample, index in enumerate(sample_data)}

    def _write_abnormal_instances_menas(
        self,
        dataset_translation_diffs: Dict[str, Dict[str, Dict[str, List[tuple]]]],
        means: Dict[str, List[tuple]],
        sample_to_frame_mapping: Dict[str, Dict[str, int]],
    ) -> None:
        """ """
        # Move to scene_token: instance: sample: translation_diff
        dataset_instance_sample_diffs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for scene_token, scene_data in dataset_translation_diffs.items():
            for sample_token, sample_data in scene_data.items():
                for instance_name, translation_diffs in sample_data.items():
                    dataset_instance_sample_diffs[scene_token][instance_name][sample_token] = translation_diffs

        columns = ["t4dataset", "instance_token", "instance_name"] + [f"frame_{i+1}" for i in range(30)]
        data = []
        # Gather translation differences for each instance
        for scene_token, scene_data in dataset_instance_sample_diffs.items():
            for instance_name, instance_data in scene_data.items():
                frames = [False] * 30
                # Extract the category name from the instance name
                category_name, instance_token, name = instance_name.split("/")
                category_thresholds = means[category_name]
                instance_row = [scene_token, instance_token, name]
                for sample_token, translation_diff in instance_data.items():
                    sample_frame_index = sample_to_frame_mapping[scene_token][sample_token]
                    dist_threshold = category_thresholds[3][0] + category_thresholds[3][1] * 2.5
                    if translation_diff[-1] > dist_threshold:
                        frames[sample_frame_index] = True
                        frames[sample_frame_index + 1] = True

                if any(frames):
                    instance_row += frames
                    data.append(instance_row)
        # Write to CSV
        csv_file_name = self.full_output_path / "translation_diff_abnormal_instances_means.csv"
        with open(csv_file_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            for row in data:
                writer.writerow(row)
        print_log(f"Saved translation diff plot to {csv_file_name}")

    def _write_abnormal_instances(
        self,
        dataset_translation_diffs: Dict[str, Dict[str, Dict[str, List[tuple]]]],
        iqrs: Dict[str, List[tuple[float]]],
        sample_to_frame_mapping: Dict[str, Dict[str, int]],
    ) -> None:
        """ """
        # Move to scene_token: instance: sample: translation_diff
        dataset_instance_sample_diffs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for scene_token, scene_data in dataset_translation_diffs.items():
            for sample_token, sample_data in scene_data.items():
                for instance_name, translation_diffs in sample_data.items():
                    dataset_instance_sample_diffs[scene_token][instance_name][sample_token] = translation_diffs

        columns = ["t4dataset", "instance_token", "instance_name"] + [f"frame_{i+1}" for i in range(30)]
        data = []
        # Gather translation differences for each instance
        for scene_token, scene_data in dataset_instance_sample_diffs.items():
            for instance_name, instance_data in scene_data.items():
                frames = [False] * 30
                # Extract the category name from the instance name
                category_name, instance_token, name = instance_name.split("/")
                category_thresholds = iqrs[category_name]
                instance_row = [scene_token, instance_token, name]
                for sample_token, translation_diff in instance_data.items():
                    dist_threshold = category_thresholds[3][0] + category_thresholds[3][1] * 2.5
                    sample_frame_index = sample_to_frame_mapping[scene_token][sample_token]
                    if translation_diff[-1] > dist_threshold:
                        frames[sample_frame_index] = True
                        frames[sample_frame_index + 1] = True

                if any(frames):
                    instance_row += frames
                    data.append(instance_row)
        # Write to CSV
        csv_file_name = self.full_output_path / "translation_diff_abnormal_instances.csv"
        with open(csv_file_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            for row in data:
                writer.writerow(row)
        print_log(f"Saved translation diff plot to {csv_file_name}")

    def gather_dataset_category_translation_diff(
        self, dataset_translation_diffs: Dict[str, Dict[str, Dict[str, tuple]]]
    ) -> Dict[str, List[tuple]]:
        """
        :param dataset_translation_diffs: {scene: {sample: {instance_name: translation_diff}}}.
        :return: {category_name: [translation_diff]}.
        """
        category_translation_diffs = defaultdict(list)
        # Gather translation differences for each instance
        for scene_token, scene_data in dataset_translation_diffs.items():
            for sample_token, sample_data in scene_data.items():
                for instance_name, translation_diffs in sample_data.items():
                    # Extract the category name from the instance name
                    category_name = instance_name.split("/")[0]
                    category_translation_diffs[category_name].append(translation_diffs)

        return category_translation_diffs

    def plot_dataset_translation_diff(
        self,
        dataset_name: str,
        category_translation_diffs: Dict[str, List[tuple]],
        figsize: tuple[int, int] = (16, 16),
    ) -> Dict[str, List[tuple[float]]]:
        """
        :param category_translation_diffs: {category_name: [translation_diff]}.
        """
        translation_names = ["X", "Y", "Z", "Dist"]
        iqrs = defaultdict(list)
        for category_name, translation_diffs in category_translation_diffs.items():
            # Plot translation differences for each category and differences in translations
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize)
            axes = axes.flatten()
            for index in range(4):
                translation_diff = [diff[index] for diff in translation_diffs]
                ax = axes[index]
                translation_name = translation_names[index]
                ax.boxplot(translation_diff, vert=True, patch_artist=True)

                # Compute quartiles and IQR
                q1 = np.percentile(translation_diff, 25)
                q3 = np.percentile(translation_diff, 75)
                median = np.percentile(translation_diff, 50)
                iqr = q3 - q1
                iqrs[category_name].append((q3, iqr))
                mean = np.mean(translation_diff)
                std = np.std(translation_diff)

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
                ax.set_ylabel("Difference")
                ax.set_xticks([1])
                ax.set_xticklabels([translation_name])
                ax.legend()

            # Save the plot
            plot_file_name = self.full_output_path / self.analysis_file_name.format(category_name, dataset_name)
            fig.suptitle(category_name)
            plt.tight_layout()
            plt.savefig(plot_file_name)
            print_log(f"Saved translation diff plot to {plot_file_name}")
            plt.close()
        return iqrs

    def plot_dataset_translation_diff_hist(
        self,
        dataset_name: str,
        category_translation_diffs: Dict[str, List[tuple]],
        figsize: tuple[int, int] = (16, 16),
    ) -> Dict[str, List[tuple]]:
        """
        :param category_translation_diffs: {category_name: [translation_diff]}.
        """
        percentiles = [0, 25, 50, 75, 95, 98, 99, 100]
        colors = ["blue", "orange", "green", "red", "purple", "brown", "olive", "pink"]
        translation_names = ["X", "Y", "Z", "Dist"]
        means = defaultdict(list)
        for category_name, translation_diffs in category_translation_diffs.items():
            # Plot translation differences for each category and differences in translations
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize)
            axes = axes.flatten()
            for index in range(4):
                translation_diff = [diff[index] for diff in translation_diffs]
                ax = axes[index]
                translation_name = translation_names[index]

                p_values = np.percentile(translation_diff, percentiles)
                mean = np.mean(translation_diff)
                std = np.std(translation_diff)
                means[category_name].append((mean, std))

                ax.hist(translation_diff, bins=self.bins, log=True)
                for value, percentile, color in zip(p_values, percentiles, colors):
                    ax.axvline(value, color=color, linestyle="dashed", linewidth=2, label=f"P{percentile}:{value:.2f}")

                ax.axvline(
                    mean, color="black", linestyle="dashed", linewidth=2, label=f"mean:{mean:.2f} (std:{std:.2f})"
                )
                ax.set_ylabel("Frequency")
                ax.set_xlabel("Differences")
                ax.set_title(translation_name)
                ax.legend(loc=self.legend_loc)
                # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            # Save the plot
            plot_file_name = self.full_output_path / self.analysis_bin_file_name.format(category_name, dataset_name)
            fig.suptitle(category_name)
            plt.tight_layout()
            plt.savefig(plot_file_name)
            print_log(f"Saved translation diff plot to {plot_file_name}")
            plt.close()
        return means

    def compute_sceneario_trans_diff(self, scenario_data: ScenarioData) -> Dict[str, Dict[str, tuple]]:
        """Compute translation difference between two frames."""
        # sample_token: instance_name: []
        instance_trans_diffs: Dict[str, Dict[str, tuple]] = defaultdict(lambda: defaultdict(tuple))

        sample_data = sorted(scenario_data.sample_data.values(), key=lambda x: x.timestamp)

        # Convert box data to {sample: instance_token: box}
        sample_instance_box: Dict[str, Dict[str, Detection3DBox]] = {
            sample.sample_token: {
                detection_3d_box.box.uuid: detection_3d_box for detection_3d_box in sample.detection_3d_boxes
            }
            for sample in sample_data
        }

        for sample in sample_data:
            sample_token = sample.sample_token
            if sample.next_sample_token is None:
                continue

            for detection_3d_box in sample.detection_3d_boxes:
                box_category_name = detection_3d_box.box.semantic_label.name
                if self.remapping_classes is not None:
                    # If no category found from the remapping, then it uses the original category name
                    box_category_name = self.remapping_classes.get(box_category_name, box_category_name)

                # Get the next instance
                next_sample = sample_instance_box.get(sample.next_sample_token, None)
                if next_sample is None:
                    continue

                next_instance_box: Detection3DBox = next_sample.get(detection_3d_box.box.uuid, None)
                if next_instance_box is None:
                    continue

                instance_name = f"{box_category_name}/{detection_3d_box.box.uuid}/{detection_3d_box.instance_name}"
                # Compute translation difference in x, y, z
                current_x, current_y, current_z = detection_3d_box.box.position
                next_x, next_y, next_z = next_instance_box.box.position
                dist = np.sqrt((current_x - next_x) ** 2 + (current_y - next_y) ** 2 + (current_z - next_z) ** 2)
                translation_diff = (abs(current_x - next_x), abs(current_y - next_y), abs(current_z - next_z), dist)

                instance_trans_diffs[sample_token][instance_name] = translation_diff

        return instance_trans_diffs

    def run(self, dataset_split_analysis_data: Dict[DatasetSplitName, AnalysisData]) -> None:
        """Inherited, check the superclass."""
        print_log(f"Running {self.__class__.__name__}")
        # Convert to {dataset: AnalysisData}
        dataset_analysis_data: Dict[str, List[AnalysisData]] = defaultdict(list)
        for dataset_split_name, analysis_data in dataset_split_analysis_data.items():
            dataset_analysis_data[dataset_split_name.dataset_version].append(analysis_data)

        for dataset_name, analysis_data in dataset_analysis_data.items():
            scene_trans_diff = defaultdict(lambda: defaultdict(lambda: defaultdict(tuple)))
            sccene_sampling_to_timestamp_mapping = defaultdict(defaultdict(int))
            for analysis in analysis_data:
                # scene: sample: instance
                for scene_token, scenario_data in analysis.scenario_data.items():
                    trans_diff = self.compute_sceneario_trans_diff(
                        scenario_data=scenario_data,
                    )
                    scene_trans_diff[scene_token] = trans_diff
                    # Compute mapping sample to frame index
                    mapping_sample_to_frame_index = self.compute_mapping_sample_to_frame_index(
                        scenario_data=scenario_data
                    )
                    sccene_sampling_to_timestamp_mapping[scene_token] = mapping_sample_to_frame_index

            category_translation_diffs = self.gather_dataset_category_translation_diff(scene_trans_diff)
            dataset_version = dataset_split_name.dataset_version
            iqrs = self.plot_dataset_translation_diff(
                dataset_name=dataset_version,
                category_translation_diffs=category_translation_diffs,
            )
            # Write abnormal instances
            self._write_abnormal_instances(
                dataset_translation_diffs=scene_trans_diff,
                iqrs=iqrs,
                sample_to_frame_mapping=sccene_sampling_to_timestamp_mapping,
            )
            means = self.plot_dataset_translation_diff_hist(
                dataset_name=dataset_version,
                category_translation_diffs=category_translation_diffs,
            )
            self._write_abnormal_instances_menas(
                dataset_translation_diffs=scene_trans_diff,
                means=means,
                sample_to_frame_mapping=sccene_sampling_to_timestamp_mapping,
            )
        print_log(f"Done running {self.__class__.__name__}")
