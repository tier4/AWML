from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset
from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class T4CalibrationClassificationDataset(BaseDataset):
    """Dataset for T4 Calibration Classification using the info.pkl structure.
    This dataset loads calibration classification data from T4 dataset format.
    Each sample contains comprehensive sensor and calibration information including
    camera images, lidar point clouds, and geometric transformations.
    Args:
        ann_file (str): Path to the annotation file (info.pkl) containing sample data.
        pipeline (list, optional): Data processing pipeline to apply to each sample.
        data_root (str, optional): Root directory for data files.
        indices (int or Sequence[int], optional): If set, only use the specified indices or first N samples.
        lazy_init (bool, optional): Whether to lazy initialize the dataset.
        serialize_data (bool, optional): Whether to serialize data for memory efficiency.
        filter_cfg (dict, optional): Config for filtering data.
    """

    def __init__(
        self,
        ann_file: str,
        pipeline: Optional[List[Union[Dict[str, Any], Callable]]] = None,
        data_root: Optional[str] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        lazy_init: bool = False,
        serialize_data: bool = True,
        filter_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the T4 Calibration Classification Dataset.
        Args:
            ann_file (str): Path to the annotation file (info.pkl) containing sample data.
            pipeline (list, optional): Data processing pipeline to apply to each sample.
            data_root (str, optional): Root directory for data files, passed to pipeline transforms.
            indices (int or Sequence[int], optional): If set, only use the specified indices or first N samples.
            lazy_init (bool, optional): Whether to lazy initialize the dataset.
            serialize_data (bool, optional): Whether to serialize data for memory efficiency.
            filter_cfg (dict, optional): Config for filtering data.
        """
        # Handle pipeline configuration for CalibrationClassificationTransform
        if pipeline:
            pipeline = [
                (
                    dict(x)
                    if not (isinstance(x, dict) and x.get("type") == "CalibrationClassificationTransform")
                    else {**x, "data_root": data_root}
                )
                for x in pipeline
            ]

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            pipeline=pipeline,
            indices=indices,
            lazy_init=lazy_init,
            serialize_data=serialize_data,
            filter_cfg=filter_cfg,
        )

    def parse_data_info(self, raw_data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw annotation to target format for T4 dataset.
        T4 dataset has a different structure where image paths are nested under
        'images' dictionary for each camera, rather than at the root level.
        Args:
            raw_data_info (dict): Raw data information from T4 dataset.
        Returns:
            dict: Parsed data information.
        """
        # For T4 dataset, we don't need to process img_path at root level
        # since the structure is different. Just return the raw data as is.
        return raw_data_info

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index. Required by BaseDataset.
        For calibration classification task, the label is generated dynamically
        in the transform pipeline (CalibrationClassificationTransform).
        Since the label is not available in the raw data, we return a default
        category ID list that represents the possible classes (0: miscalibrated, 1: correctly calibrated).
        Args:
            idx (int): The index of data.
        Returns:
            list[int]: Category IDs for the sample. For calibration classification,
                      this returns [0, 1] representing the two possible classes.
        """
        # For calibration classification, labels are generated dynamically in the transform
        # We return both possible class IDs (0: miscalibrated, 1: correctly calibrated)
        # This allows ClassBalancedDataset and other wrappers to work properly
        return [0, 1]

    def prepare_data(self, idx: int) -> Any:
        """Get data processed by ``self.pipeline``.
        Args:
            idx (int): The index of data.
        Returns:
            Any: Processed data.
        """
        sample = self.get_data_info(idx)
        sample = self.pipeline(sample)
        return sample
