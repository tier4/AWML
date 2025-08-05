from abc import ABC
from pathlib import Path
from typing import Any, Dict

import mmengine


class DatasetPreparationBase:

    def __init__(self, root_path: Path, config: Any, info_save_path: Path, info_version: str) -> None:
        """
        Base class of dataset prepation.
        :param root_path: Root path that contains data.
        :param config: Configuration for the dataset prepration.
        :param info_save_path: Path to save a dictionary of dataset information.
        :param info_version: Version name for dataset information.
        """
        self.root_path = root_path
        self.config = config
        self.info_save_path = info_save_path
        self.info_version = info_version

        # Make the output path
        self.info_save_path.mkdirs(exist_ok=True, parents=True)

    def run(self) -> None:
        """
        Run dataset preparation to convert dataset to corresponding info format.
        """
        raise NotImplementedError

    def save_info_file(self, info: Dict[str, Any], info_file_name: str) -> None:
        """
        Save a dictionary of datasets information to pickle file that is used by downstream tasks later.
        :param info: Selected info from datasets.
        :param info_file_name: Info output file name.
        """
        info_file_save_path = self.info_save_path / info_file_name
        mmengine.dump(info, info_file_save_path)
