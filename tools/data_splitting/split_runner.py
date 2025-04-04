from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from mmengine.config import Config
from mmengine.logging import print_log
from t4_devkit import Tier4
from t4_devkit.common.timestamp import us2sec

from tools.analysis_3d.data_classes import AnalysisData, DatasetSplitName, SampleData, ScenarioData
from tools.analysis_3d.split_options import SplitOptions
from tools.analysis_3d.utils import extract_tier4_sample_data
from tools.detection3d.create_data_t4dataset import get_scene_root_dir_path


class SplitRunner:
    """Runner to run list of analyses for the selected dataset."""

    def __init__(
        self,
        data_root_path: str,
        config_path: str,
        out_path: str,
        rewrite_scenario_data: bool = False,
    ) -> None:
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
        self.rewrite_scenario_data = rewrite_scenario_data
        self.parquet_file_path = self.out_path / "analysis_data_cache.parquet"

    def _get_dataset_scenario_names(self, dataset_version: str) -> Dict[str, List[str]]:
        """
        Get list of scenarios names for different splits in a dataset.
        :return: A dict of {split name: [scenario names in a split]}.
        """
        dataset_yaml_file = Path(self.config.dataset_version_config_root) / (dataset_version + ".yaml")
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
                sample_token=sample.token,
                boxes=tier4_sample_data.boxes,
                timestamp=tier4_sample_data.sd_record.timestamp,
                ego_pose_translation=tier4_sample_data.pose_record.translation,
            )
        return sample_data

    def _extra_scenario_data(
        self,
        dataset_version: str,
        scene_tokens: List[str],
    ) -> Dict[str, ScenarioData]:
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

            t4 = Tier4(version="annotation", data_root=str(scene_root_dir_path), verbose=False)
            sample_data = self._extract_sample_data(t4=t4)
            scenario_data[scene_token] = ScenarioData(scene_token=scene_token, sample_data=sample_data)
        return scenario_data

    def _save_to_parquet(self, df: pd.DataFrame) -> None:
        """
        Save the analysis data to pickle.
        :param analysis_data: A dict of {dataset split name: AnalysisData}.
        """
        # Save the DataFrame to a Parquet file
        print_log(f"Saving analysis data to {self.parquet_file_path}")
        df.to_parquet(self.parquet_file_path, index=False)

    def _read_parquet(self) -> pd.DataFrame:
        """
        Save the analysis data to pickle.
        :param analysis_data: A dict of {dataset split name: AnalysisData}.
        """
        # Save the DataFrame to a Parquet file
        print_log(f"Reading {self.parquet_file_path}")
        df = pd.read_parquet(self.parquet_file_path)
        return df

    def _compute_dataframe_category_counts(self, analysis_data: AnalysisData) -> Dict[str, Dict[str, int]]:
        """ """
        category_counts = {}
        for scenario_token, scenario in analysis_data.scenario_data.items():
            counts = defaultdict(int)
            for sample in scenario.sample_data.values():
                for bbox in sample.detection_3d_boxes:
                    category = f"category_{bbox.box.semantic_label.name}"
                    counts[category] += 1
            category_counts[scenario_token] = counts

        all_categories = set()
        for counts in category_counts.values():
            all_categories.update(counts.keys())

        for counts in category_counts.values():
            for category in all_categories:
                if category not in counts:
                    counts[category] = 0

        return category_counts

    def _compute_dataframe_starting_timestamp(self, analysis_data: AnalysisData) -> Dict[str, int]:
        """ """
        starting_timestamps = {}
        for scenario_token, scenario in analysis_data.scenario_data.items():
            starting_timestamp = min([sample.timestamp for sample in scenario.sample_data.values()])
            starting_timestamp_in_seconds: float = us2sec(starting_timestamp)
            starting_timestamp_int: int = int(starting_timestamp_in_seconds)
            starting_timestamps[scenario_token] = starting_timestamp_int

        return starting_timestamps

    def _create_scenario_dataframe(self, analysis_data: AnalysisData) -> pd.DataFrame:
        """ """
        category_counts = self._compute_dataframe_category_counts(analysis_data=analysis_data)
        starting_timestamps = self._compute_dataframe_starting_timestamp(analysis_data=analysis_data)

        data = []
        for scenario_token, counts in category_counts.items():
            data.append(
                {
                    "scenario_token": scenario_token,
                    **counts,
                    "starting_timestamp": starting_timestamps[scenario_token],
                }
            )
        print_log(f"Created DataFrame with {len(data)} rows")
        return pd.DataFrame(data)

    def run(self) -> pd.DataFrame:
        """Run the AnalysisRunner."""
        print_log("Running SplitRunner...")
        if not self.rewrite_scenario_data and self.parquet_file_path.exists():
            scenario_df = self._read_parquet()
        else:
            # Create dataset split names
            dataset_split_names = {
                DatasetSplitName(split_name=split_option.value, dataset_version=dataset_version)
                for split_option in SplitOptions
                for dataset_version in self.config.dataset_version_list
            }

            dataset_scenario_names = {
                dataset_version: self._get_dataset_scenario_names(dataset_version=dataset_version)
                for dataset_version in self.config.dataset_version_list
            }
            all_scene_tokens = []
            for dataset_split_name in dataset_split_names:
                dataset_version = dataset_split_name.dataset_version
                split_name = dataset_split_name.split_name

                print_log(f"Creating analyses for dataset: {dataset_version} split: {split_name}", logger="current")
                scene_tokens = dataset_scenario_names[dataset_version].get(split_name, None)
                if scene_tokens is None:
                    raise ValueError(f"{split_name} does not exist in the {dataset_version}.yaml!")
                all_scene_tokens += scene_tokens

            scenario_data = self._extra_scenario_data(scene_tokens=all_scene_tokens, dataset_version=dataset_version)
            analysis_data = AnalysisData(
                data_root_path=self.data_root_path, dataset_version=dataset_version, scenario_data=scenario_data
            )
            scenario_df = self._create_scenario_dataframe(analysis_data=analysis_data)
            self._save_to_parquet(df=scenario_df)
        return scenario_df
