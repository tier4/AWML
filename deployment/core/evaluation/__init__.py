"""Evaluation subpackage for deployment core."""

from deployment.core.evaluation.base_evaluator import BaseEvaluator
from deployment.core.evaluation.evaluator_types import (
    EvalResultDict,
    ModelSpec,
    VerifyResultDict,
)
from deployment.core.evaluation.output_comparator import (
    OutputComparator,
    OutputDiffSummary,
    TensorDiffDetail,
)
from deployment.core.evaluation.verification_runner import (
    SampleVerificationResult,
    VerificationHooks,
    VerificationRunner,
)

__all__ = [
    "BaseEvaluator",
    "OutputDiffSummary",
    "TensorDiffDetail",
    "EvalResultDict",
    "ModelSpec",
    "OutputComparator",
    "SampleVerificationResult",
    "VerificationHooks",
    "VerificationRunner",
    "VerifyResultDict",
]
