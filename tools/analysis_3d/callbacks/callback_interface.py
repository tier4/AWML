from abc import ABC
from typing import Dict

from tools.analysis_3d.data_classes import AnalysisData, DatasetSplitName


class AnalysisCallbackInterface(ABC):

    def run(
        self,
        dataset_split_analysis_data: Dict[DatasetSplitName, AnalysisData],
    ) -> None:
        """
        Run analysis callback with the given analysis data for every dataset.
        :param dataset_split_analysis_data: Analysis data for every dataset and splits. A dict
            of {DatasetSplitName: AnalysisData}.
        """
        raise NotImplementedError
