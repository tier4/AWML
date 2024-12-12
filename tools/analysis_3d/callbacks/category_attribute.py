from pathlib import Path
from typing import Dict, Optional

from mmengine.logging import print_log

from tools.analysis_3d.callbacks.category import CategoryAnalysisCallback
from tools.analysis_3d.data_classes import AnalysisData, DatasetSplitName
from tools.analysis_3d.split_options import SplitOptions


class CategoryAttributeAnalysisCallback(CategoryAnalysisCallback):
    """Compute frequency of attributes for the selected category."""

    def __init__(
        self,
        out_path: Path,
        category_name: str,
        analysis_dir: str,
        remapping_classes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        :param out_path: Path to save outputs.
        :param category_name: Selected category to compute the frequency.
        :param analysis_dir: Folder name to save outputs.
        :param remapping_classes: Set if select the category after remapping.
        """
        super(CategoryAttributeAnalysisCallback, self).__init__(
            out_path=out_path, analysis_dir=analysis_dir, remapping_classes=remapping_classes
        )
        self.category_name = category_name
        self.analysis_file_name = self.category_name + "_attr_count_{}.png"
        self.y_axis_label = f"Attributes in {category_name}"
        self.x_axis_label = f"Attributes counts in {category_name} by datasets"

    def run(self, dataset_split_analysis_data: Dict[DatasetSplitName, AnalysisData]) -> None:
        """Inherited, check the superclass."""
        print_log(f"Running {self.__class__.__name__}")
        for split_option in SplitOptions:
            dataset_category_counts = {}
            for dataset_split_name, analysis_data in dataset_split_analysis_data.items():
                split_name = dataset_split_name.split_name
                if split_name != split_option:
                    continue

                dataset_name = dataset_split_name.dataset_version
                dataset_category_counts[dataset_name] = analysis_data.aggregate_category_attr_counts(
                    remapping_classes=self.remapping_classes, category_name=self.category_name
                )

            self._visualize_total_category_counts(
                dataset_category_counts=dataset_category_counts, split_name=split_option.value
            )
        print_log(f"Done running {self.__class__.__name__}")
