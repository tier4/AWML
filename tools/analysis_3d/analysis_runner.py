from pathlib import Path
from typing import Dict, List

import yaml
from mmengine.config import Config
from mmengine.logging import print_log

from tools.analysis_3d.callbacks.callback_interface import AnalysisCallbackInterface
from tools.analysis_3d.callbacks.category import CategoryAnalysisCallback
from tools.analysis_3d.callbacks.category_attribute import CategoryAttributeAnalysisCallback
from tools.analysis_3d.data_classes import AnalysisData, ScenarioData, SampleData, DatasetSplitName
from tools.analysis_3d.split_options import SplitOptions
from tools.analysis_3d.utils import extract_tier4_sample_data
from tools.detection3d.create_data_t4dataset import get_scene_root_dir_path
from t4_devkit import Tier4


class AnalysisRunner:
    """ Runner to run list of analyses for the selected dataset. """

    def __init__(self, data_root_path: str, config_path: str,
                 out_path: str) -> None:
        """
        :param data_root_path: Path where to save data.
        :param config_path: Configuration path for a dataset.
        :param out_path: Path where to save output.
        """
        self.data_root_path = data_root_path
        self.config_path = config_path
        self.out_path = Path(out_path)
        # Initialization
        self.config = Config.fromfile(self.config_path)
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.remapping_classes = self.config.name_mapping

        # Default callbacks to generate analyses
        # TODO (KokSeang): Configure through CLI
        self.analysis_callbacks: List[AnalysisCallbackInterface] = [
            CategoryAnalysisCallback(
                out_path=self.out_path,
                remapping_classes=self.remapping_classes),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name='vehicle.motorcycle',
                analysis_dir='vehicle_motorcycle_attr'),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name='vehicle.bicycle',
                analysis_dir='vehicle_bicycle_attr'),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name='bicycle',
                analysis_dir='bicycle_attr'),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name='motorcycle',
                analysis_dir='motorcycle_attr'),
            CategoryAttributeAnalysisCallback(
                out_path=self.out_path,
                category_name='bicycle',
                analysis_dir='remapping_bicycle_attr',
                remapping_classes=self.remapping_classes),
        ]

    def _get_dataset_scenario_names(
            self, dataset_version: str) -> Dict[str, List[str]]:
        """
        Get list of scenarios names for different splits in a dataset.
        :return: A dict of {split name: [scenario names in a split]}. 
        """
        dataset_yaml_file = Path(self.config.dataset_version_config_root) / (
            dataset_version + ".yaml")
        with open(dataset_yaml_file, "r") as f:
            dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)
            return dataset_list_dict

    def _extract_sample_data(self, t4: Tier4) -> Dict[str, SampleData]:
        """
        Extract data for every sample.
        :param t4: Tier4 interface.
        :return: A dict of {sample token: SampleData}.
        """
        sample_data = {}
        for sample in t4.sample:
            # Extract sample data
            tier4_sample_data = extract_tier4_sample_data(sample=sample, t4=t4)

            # Convert to SampleData
            sample_data[sample.token] = SampleData.create_sample_data(
                sample_token=sample.token, boxes=tier4_sample_data.boxes)
        return sample_data

    def _extra_scenario_data(
            self, dataset_version: str,
            scene_tokens: List[str]) -> Dict[str, ScenarioData]:
        """
        Extra data for every scenario.
        :param dataset_version: Dataset version.
        :param scene_tokens: List of scenario tokens in the dataset version.
        :return: A dict of {scenario token: ScenarioData}.
        """
        scenario_data = {}
        for scene_token in scene_tokens:
            print_log(f"Creating scenario data for the scene: {scene_token}")
            scene_root_dir_path = get_scene_root_dir_path(
                root_path=self.data_root_path,
                dataset_version=dataset_version,
                scene_id=scene_token,
            )
            scene_root_dir_path = Path(scene_root_dir_path)
            if not scene_root_dir_path.is_dir():
                raise ValueError(f"{scene_root_dir_path} does not exist.")

            t4 = Tier4(
                version="annotation",
                data_root=str(scene_root_dir_path),
                verbose=False)
            sample_data = self._extract_sample_data(t4=t4)
            scenario_data[scene_token] = ScenarioData(
                scene_token=scene_token, sample_data=sample_data)
        return scenario_data

    def run(self) -> None:
        """ Run the AnalysisRunner. """
        print_log("Running AnalysesRunner...")
        # Create dataset split names
        dataset_split_names = {
            DatasetSplitName(
                split_name=split_option.value, dataset_version=dataset_version)
            for split_option in SplitOptions
            for dataset_version in self.config.dataset_version_list
        }

        dataset_scenario_names = {
            dataset_version:
            self._get_dataset_scenario_names(dataset_version=dataset_version)
            for dataset_version in self.config.dataset_version_list
        }
        dataset_split_analysis_data = {}
        for dataset_split_name in dataset_split_names:
            dataset_version = dataset_split_name.dataset_version
            split_name = dataset_split_name.split_name

            print_log(
                f"Creating analyses for dataset: {dataset_version} split: {split_name}",
                logger="current")
            scene_tokens = dataset_scenario_names[dataset_version].get(
                split_name, None)
            if scene_tokens is None:
                raise ValueError(
                    f"{split_name} does not exist in the {dataset_version}.yaml!"
                )

            scenario_data = self._extra_scenario_data(
                scene_tokens=scene_tokens, dataset_version=dataset_version)

            analysis_data = AnalysisData(
                data_root_path=self.data_root_path,
                dataset_version=dataset_version,
                scenario_data=scenario_data)
            dataset_split_analysis_data[dataset_split_name] = analysis_data

        print_log("=========")
        print_log("Calling Analysis Callbacks")
        for callback in self.analysis_callbacks:
            callback.run(
                dataset_split_analysis_data=dataset_split_analysis_data)
        print_log("=========")
        print_log("Done running AnalysisRunner!!")
