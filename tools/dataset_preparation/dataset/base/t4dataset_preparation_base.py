from pathlib import Path
from typing import Dict, Any, List

import yaml 
from mmengine.logging import print_log
from t4_devkit import Tier4

from tools.dataset_preparation.dataset.base.dataset_preparation_base import DatasetPreparationBase
from tools.dataset_preparation.enum import DatasetInfoSplitKey
from tools.detection3d.create_data_t4dataset import get_scene_root_dir_path 

class T4DatasetPreparationBase(DatasetPreparationBase):
    
    def __init__(self,
                 root_path: Path,
                 config: Any, 
                 info_save_path: Path, 
                 info_version: str,
                 use_available_dataset_version: bool = False
                 ) -> None: 
        """
        Base class of dataset prepation. 
        :param config: Configuration for the dataset prepration.
        """
        super(T4DatasetPreparationBase, self).__init__(
            root_path=root_path,
            config=config, 
            info_save_path=info_save_path, 
            info_version=info_version
        )
        self.use_available_dataset_version = use_available_dataset_version
        self.t4dataset_info_file_template = "t4dataset_{}_infos_{}.pkl"

    def process_t4dataset(self, t4_dataset: Tier4) -> Dict[str, Any]:
        """
        Process a t4dataset and prepare it usable format to the AWML framework.
        :param t4_dataset: Tier4 data object for a t4dataset.
        :return: A dict of {frame identifier: frame data}.
        """
        # For the base case, it does nothing.
        raise NotImplementedError
    
    def save_t4_info_file(self, info: Dict[str, Any], split_name: str):
        """
        Save t4 infos to a file.
        :param infos: Selected T4 info.  
        """        
        info_split_file_name = self.t4dataset_info_file_template.format(self.info_version, split_name)
        self.save_info_file(info=info, info_file_name=info_split_file_name)

    def extract_metainfo(self) -> Dict[str, Any]:
        """
        Extract metainfo.
        """
        return {}

    def run(
        self,
    ) -> None:
        """
        Run dataset preparation to convert dataset to corresponding info format. 
        """
        data_info = {
            DatasetInfoSplitKey.TRAIN: [],
            DatasetInfoSplitKey.VAL: [],
            DatasetInfoSplitKey.TEST: [],
        }
        metainfo = self.extract_metainfo()

        for dataset_version in self.config.dataset_version_list:
            dataset_list = Path(self.config.dataset_version_config_root) / (dataset_version + ".yaml")
            with open(dataset_list, "r") as f:
                dataset_list_dict: Dict[str, List[str]] = yaml.safe_load(f)

            for split in [DatasetInfoSplitKey.TRAIN, DatasetInfoSplitKey.VAL, DatasetInfoSplitKey.TEST]:
                print_log(f"Creating data info for split: {split}", logger="current")
                for scene_id in dataset_list_dict.get(split, []):
                    print_log(f"Creating data info for scene: {scene_id}")

                    t4_dataset_id, t4_dataset_version_id = scene_id.split("/")
                    scene_root_dir_path = Path(self.root_path) / dataset_version / t4_dataset_id / t4_dataset_version_id
                    if not scene_root_dir_path.exists():
                        if self.use_available_dataset_version:
                            print_log(
                                "Warning: The version of the dataset specified in the config file does not exist. " \
                                "Will use whatever is available locally."
                            )
                            scene_root_dir_path = get_scene_root_dir_path(self.root_path, dataset_version, t4_dataset_id)
                        else:
                            raise ValueError(f"{scene_root_dir_path} does not exist.")

                    t4_dataset = Tier4(
                        version="annotation",
                        data_root=scene_root_dir_path,
                        verbose=False,
                    )
                    
                    info = self.process_t4dataset(
                        t4_dataset=t4_dataset
                    )

                    data_info[split].extend(info.values())
        
        info_pairs = {
            DatasetInfoSplitKey.TRAIN: data_info[DatasetInfoSplitKey.TRAIN],
            DatasetInfoSplitKey.VAL: data_info[DatasetInfoSplitKey.VAL],
            DatasetInfoSplitKey.TEST: data_info[DatasetInfoSplitKey.TEST],
            DatasetInfoSplitKey.TRAIN_VAL: data_info[DatasetInfoSplitKey.TRAIN] + data_info[DatasetInfoSplitKey.VAL],
            DatasetInfoSplitKey.ALL: data_info,
        }
        for split_name, info in info_pairs.items():
            format_info = {
                "data_list": info,
                "metainfo": metainfo
            }
            self.save_t4_info_file(info=format_info, split_name=split_name)
