from typing import Dict, List

from t4_devkit import Tier4

from tools.analysis_2d.data_classes import (
    SampleData2D,
)
from tools.analysis_2d.utils import extract_tier4_sample_data
from tools.analysis_3d.analysis_runner import AnalysisRunner
from tools.analysis_3d.callbacks.callback_interface import AnalysisCallbackInterface
from tools.analysis_3d.callbacks.category import CategoryAnalysisCallback
from tools.analysis_3d.callbacks.category_attribute import CategoryAttributeAnalysisCallback


class AnalysisRunner2D(AnalysisRunner):
    """Runner to run list of analyses for the selected dataset."""

    def __init__(
        self,
        data_root_path: str,
        config_path: str,
        out_path: str,
    ) -> None:
        """
        :param data_root_path: Path where to save data.
        :param config_path: Configuration path for a dataset.
        :param out_path: Path where to save output.
        """
        super().__init__(data_root_path, config_path, out_path)

        # Override remapping_classes for 2D analysis and callbacks
        self.remapping_classes = self.config.class_mappings
        self.analysis_callbacks: List[AnalysisCallbackInterface] = [
            CategoryAnalysisCallback(out_path=self.out_path, remapping_classes=self.remapping_classes),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name="green",
                analysis_dir="green_attributes",
                remapping_classes=self.remapping_classes,
            ),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name="red",
                analysis_dir="red_attributes",
                remapping_classes=self.remapping_classes,
            ),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name="yellow",
                analysis_dir="yellow_attributes",
                remapping_classes=self.remapping_classes,
            ),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name="left,red",
                analysis_dir="left_red_attributes",
                remapping_classes=self.remapping_classes,
            ),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name="red,up_left",
                analysis_dir="red_up_left_attributes",
                remapping_classes=self.remapping_classes,
            ),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name="red,right",
                analysis_dir="red_right_attributes",
                remapping_classes=self.remapping_classes,
            ),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name="red,straight",
                analysis_dir="red_straight_attributes",
                remapping_classes=self.remapping_classes,
            ),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name="left,red,straight",
                analysis_dir="left_red_straight_attributes",
                remapping_classes=self.remapping_classes,
            ),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name="unknown",
                analysis_dir="unknown_attributes",
                remapping_classes=self.remapping_classes,
            ),
        ]

    def _extract_sample_data(self, t4: Tier4) -> Dict[str, SampleData2D]:
        """
        Extract data for every sample.
        :param t4: Tier4 interface.
        :return: A dict of {sample token: SampleData}.
        """
        sample_data = {}
        for sample in t4.sample:
            # Extract sample data
            tier4_sample_data_list = extract_tier4_sample_data(sample=sample, t4=t4)

            # Convert to SampleData
            if tier4_sample_data_list is None:
                continue
            for tier4_sample_data in tier4_sample_data_list:
                sample_data[f"{sample.token},{tier4_sample_data.camera_name}"] = SampleData2D.create_sample_data(
                    sample_token=sample.token,
                    camera_name=tier4_sample_data.camera_name,
                    boxes=tier4_sample_data.boxes,
                )
        return sample_data
