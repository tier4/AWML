from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from mmdet3d.registry import METRICS
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)

from autoware_ml.detection3d.evaluation.t4metric.t4metric import T4Metric

__all__ = ["T4MetricV2"]


# [TODO] This class will refactor. We will rewrite T4Metrics
# using [autoware_perception_evaluation](https://github.com/tier4/autoware_perception_evaluation).
@METRICS.register_module()
class T4MetricV2(T4Metric):
    """T4 format evaluation metric V2."""

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        perception_evaluator_configs: Dict[str, Any],
        critical_object_filter_config: Dict[str, Any],
        frame_pass_fail_config: Dict[str, Any],
        filter_attributes: Optional[List[Tuple[str, str]]] = None,
        metric: Union[str, List[str]] = "bbox",
        modality: dict = dict(use_camera=False, use_lidar=True),
        prefix: Optional[str] = None,
        format_only: bool = False,
        jsonfile_prefix: Optional[str] = None,
        eval_version: str = "detection_cvpr_2019",
        collect_device: str = "cpu",
        backend_args: Optional[dict] = None,
        class_names: List[str] = [],
        eval_class_range: Dict[str, int] = dict(),
        name_mapping: Optional[dict] = None,
        version: str = "",
    ) -> None:
        """
        Args:
            data_root (str):
                Path of dataset root.
            ann_file (str):
                Path of annotation file.
            filter_attributes (str)
                Filter out GTs with certain attributes. For example, [['vehicle.bicycle',
                'vehicle_state.parked']].
            metric (str or List[str]):
                Metrics to be evaluated. Defaults to 'bbox'.
            modality (dict):
                Modality to specify the sensor data used as input.
                Defaults to dict(use_camera=False, use_lidar=True).
            prefix (str, optional):
                The prefix that will be added in the metric
                names to disambiguate homonymous metrics of different evaluators.
                If prefix is not provided in the argument, self.default_prefix will
                be used instead. Defaults to None.
            format_only (bool):
                Format the output results without perform
                evaluation. It is useful when you want to format the result to a
                specific format and submit it to the test server.
                Defaults to False.
            jsonfile_prefix (str, optional):
                The prefix of json files including the
                file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Defaults to None.
            eval_version (str):
                Configuration version of evaluation.
                Defaults to 'detection_cvpr_2019'.
            collect_device (str):
                Device name used for collecting results from
                different ranks during distributed training. Must be 'cpu' or 'gpu'.
                Defaults to 'cpu'.
            backend_args (dict, optional):
                Arguments to instantiate the corresponding backend. Defaults to None.
            class_names (List[str], optional):
                The class names. Defaults to [].
            eval_class_range (Dict[str, int]):
                The range of each class
            name_mapping (dict, optional):
                The data class mapping, applied to ground truth during evaluation.
                Defaults to None.
            version (str, optional):
                The version of the dataset. Defaults to "".
        """

        super(T4MetricV2, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            metric=metric,
            modality=modality,
            prefix=prefix,
            format_only=format_only,
            jsonfile_prefix=jsonfile_prefix,
            eval_version=eval_version,
            collect_device=collect_device,
            backend_args=backend_args,
            class_names=class_names,
            eval_class_range=eval_class_range,
            name_mapping=name_mapping,
            version=version,
            filter_attributes=filter_attributes,
        )

        self.perception_evaluator_configs = PerceptionEvaluationConfig(**perception_evaluator_configs)
        self.critical_object_filter_config = CriticalObjectFilterConfig(
            evaluator_config=self.perception_evaluator_configs, **critical_object_filter_config
        )
        self.frame_pass_fail_config = PerceptionPassFailConfig(
            evaluator_config=self.perception_evaluator_configs, **frame_pass_fail_config
        )

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        super().process(data_batch=data_batch, data_samples=data_samples)
