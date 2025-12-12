from .tlr_metrics import TLRFineDetectorEvaluator
from mmseg.evaluation.metrics import IoUMetric
from mmengine.registry import METRICS
METRICS.register_module()(IoUMetric)

__all__ = ["TLRFineDetectorEvaluator", "IoUMetric"]
