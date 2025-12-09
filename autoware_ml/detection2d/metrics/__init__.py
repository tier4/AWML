from mmengine.registry import METRICS
from mmseg.evaluation.metrics import IoUMetric

from .tlr_metrics import TLRFineDetectorEvaluator

# 注册 IoUMetric
METRICS.register_module()(IoUMetric)
# __all__ = ["TLRFineDetectorEvaluator"]
__all__ = ["TLRFineDetectorEvaluator", "IoUMetric"]
