from .tlr_metrics import TLRFineDetectorEvaluator
from mmseg.evaluation.metrics import IoUMetric
from mmengine.registry import METRICS
# 注册 IoUMetric
METRICS.register_module()(IoUMetric)
# __all__ = ["TLRFineDetectorEvaluator"]
__all__ = ["TLRFineDetectorEvaluator", "IoUMetric"]
