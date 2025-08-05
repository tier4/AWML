from pathlib import Path
from typing import Dict, Any, List

from t4_devkit import Tier4

from tools.dataset_preparation.dataset.base.t4dataset_preparation_base import T4DatasetPreparationBase
from tools.detection3d.create_data_t4dataset import get_info

class T4DatasetDetection3DPreparation(T4DatasetPreparationBase):
    
    def __init__(self,
                 root_path: Path,
                 config: Any, 
                 info_save_path: Path, 
                 info_version: str,
                 max_sweeps: int,
                 use_available_dataset_version: bool = False
                 ) -> None: 
        """
        Base class of dataset prepation. 
        :param config: Configuration for the dataset prepration.
        """
        super(T4DatasetDetection3DPreparation, self).__init__(
            root_path=root_path,
            config=config, 
            info_save_path=info_save_path, 
            info_version=info_version,
            use_available_dataset_version=use_available_dataset_version
        )
        self._max_sweeps = max_sweeps

    def process_t4dataset(self, t4_dataset: Tier4) -> Dict[str, Any]:
        """
        Process a t4dataset and prepare it usable format to the AWML framework.
        :return: A dict of {split_name: list of t4dataset frames}.
        """
        infos = {}
        for i, sample in enumerate(t4_dataset.sample):
            infos[i] = get_info(cfg=self.config, t4=t4_dataset, sample=sample, i=i, max_sweeps=self._max_sweeps)
        return infos
    
    def extract_metainfo(self) -> Dict[str, Any]:
        """
        Extract metainfo.
        :return A dict of metainfo about the data prepration.
        """
        return {
            "version": self.info_version, "task_name": "3d_detection", "classes": self.config.class_names
        }
