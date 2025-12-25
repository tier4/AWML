from mmengine.registry import METRICS
from mmseg.evaluation.metrics import IoUMetric

from .tlr_metrics import TLRFineDetectorEvaluator

METRICS.register_module()(IoUMetric)

__all__ = ["TLRFineDetectorEvaluator", "IoUMetric"]
