import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from mmdet3d.registry import METRICS
from perception_eval.common import DynamicObject
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import AutowareLabel, Label
from perception_eval.common.schema import FrameID
from perception_eval.common.shape import Shape, ShapeType
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig
from perception_eval.evaluation.metrics import MetricsScoreConfig
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.result.perception_frame_config import (
    CriticalObjectFilterConfig,
    PerceptionPassFailConfig,
)
from perception_eval.evaluation.result.perception_frame_result import PerceptionFrameResult
from pyquaternion.quaternion import Quaternion

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
        evaluator_metric_configs: Dict[str, Any],
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

        dynamic_object = DynamicObject(
            unix_time=1,
            frame_id=FrameID.BASE_LINK,
            position=[1.0, 1.0, 1.0],
            orientation=Quaternion([1.0, 0.0, 0.0, 0.0]),
            shape=Shape(
                shape_type=ShapeType.BOUNDING_BOX,
                size=[1.0, 1.0, 1.0],
            ),
            velocity=[1.0, 1.0, 1.0],
            semantic_score=1.0,
            semantic_label=Label(label=AutowareLabel.CAR, name="car"),
        )
        perception_frame_result = PerceptionFrameResult(
            object_results=[
                DynamicObjectWithPerceptionResult(estimated_object=dynamic_object, ground_truth_object=None)
            ],
            frame_ground_truth=FrameGroundTruth(unix_time=1, frame_name="1", objects=[dynamic_object]),
            metrics_config=self.perception_evaluator_configs.metrics_config,
            critical_object_filter_config=self.critical_object_filter_config,
            frame_pass_fail_config=self.frame_pass_fail_config,
            target_labels=self.class_names,
            unix_time=1,
        )
        with open(self.perception_evaluator_configs.result_root_directory + "outs.pkl", "wb") as f:
            pickle.dump(perception_frame_result, f)

        with open(self.perception_evaluator_configs.result_root_directory + "outs.pkl", "rb") as f:
            a = pickle.load(f)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        super().process(data_batch=data_batch, data_samples=data_samples)
