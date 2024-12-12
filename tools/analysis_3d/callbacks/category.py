from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from mmengine.logging import print_log

from tools.analysis_3d.callbacks.callback_interface import AnalysisCallbackInterface
from tools.analysis_3d.data_classes import AnalysisData, DatasetSplitName
from tools.analysis_3d.split_options import SplitOptions


class CategoryAnalysisCallback(AnalysisCallbackInterface):
    """Compute frequency for every category."""

    def __init__(
        self,
        out_path: Path,
        analysis_dir: str = "categories",
        remapping_classes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        :param out_path: Path to save outputs.
        :param analysis_dir: Folder name to save outputs.
        :param remapping_classes: Set if compute frequency of every category after remapping.
        """
        super(CategoryAnalysisCallback, self).__init__()
        self.out_path = out_path
        self.analysis_dir = analysis_dir
        self.remapping_classes = remapping_classes

        self.full_output_path = self.out_path / self.analysis_dir
        self.full_output_path.mkdir(exist_ok=True, parents=True)

        self.analysis_file_name = "category_count_{}.png"
        self.y_axis_label = "Category"
        self.x_axis_label = "Category counts by datasets"
        self.legend_loc = "upper right"

    def _visualize_total_category_counts(
        self,
        dataset_category_counts: Dict[str, Dict[str, int]],
        split_name: str,
        log_scale: bool = True,
        figsize: tuple[int, int] = (15, 15),
    ) -> None:
        """
        Visualize frequency of every category in a horizontal bar graph.
        :param dataset_category_counts: A dict of {dataset name: {category name: total counts}}.
        :param split_name: Split name, for example, train, test, val, to visualize.
        :param log_scale: Set True to make the frequency in log-scale (power of 10).
        :param figsize: Figure size.
        """
        all_available_categories = [
            category_name
            for category_counts in dataset_category_counts.values()
            for category_name in category_counts.keys()
        ]
        all_available_categories = sorted(list(set(all_available_categories)))

        # All available dataset names
        all_available_datasets = sorted(list(set(dataset_category_counts.keys())))

        # Move data to
        # {'dataset': [category count]}
        plot_data = defaultdict(list)
        for dataset_name in all_available_datasets:
            for category_name in all_available_categories:
                category_counts = dataset_category_counts[dataset_name].get(category_name, 0)
                plot_data[dataset_name].append(category_counts)
        y = np.arange(len(all_available_categories))  # the label locations
        height = min(0.25, (1.0 / len(all_available_datasets)) - 0.05)  # the width of the bars
        multiplier = 0

        _, ax = plt.subplots(figsize=figsize)
        for dataset_name, counts in plot_data.items():
            offset = height * multiplier
            rects = ax.barh(y + offset, counts, height, label=dataset_name, log=log_scale, edgecolor="w")
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(self.y_axis_label)
        ax.set_title(self.x_axis_label)
        ax.set_yticks(y + height, all_available_categories)
        ax.legend(loc=self.legend_loc)
        ax.invert_yaxis()

        plt.tight_layout()
        analysis_file_name = self.full_output_path / self.analysis_file_name.format(split_name)
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
            dataset_category_counts = {}
            for dataset_split_name, analysis_data in dataset_split_analysis_data.items():
                split_name = dataset_split_name.split_name
                if split_name != split_option.value:
                    continue

                dataset_name = dataset_split_name.dataset_version
                dataset_category_counts[dataset_name] = analysis_data.aggregate_category_counts(
                    remapping_classes=self.remapping_classes
                )

            self._visualize_total_category_counts(
                dataset_category_counts=dataset_category_counts, split_name=split_option.value
            )
        print_log(f"Done running {self.__class__.__name__}")
