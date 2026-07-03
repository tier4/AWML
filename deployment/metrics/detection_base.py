"""
Shared base for 2D and 3D detection metrics interfaces.

2D and 3D detection both produce mAP-style metrics from perception_eval and differ only
in:
    - object conversion (image-space ROI vs 3D bounding box) -> subclass ``add_frame``,
    - whether heading metrics (APH/mAPH) apply -> ``_supports_aph``,
    - which evaluator's score feeds the summary -> ``_select_summary_score``.

Everything else (flattening a ``MetricsScore`` to a metric dict and building the
``DetectionSummary``) is shared here. Both the flat dict and the summary are derived from a
single structured pass over the ``MetricsScore`` (see ``_extract_scores``); we never parse
structure back out of the flattened string keys.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from perception_eval.evaluation.metrics import MetricsScore

from deployment.metrics.base_metrics_interface import BaseMetricsInterface

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionSummary:
    """Structured summary for detection metrics (2D/3D).

    All matching modes computed by autoware_perception_evaluation are included.
    The `mAP_by_mode` and `mAPH_by_mode` dicts contain results for each matching mode.
    """

    mAP_by_mode: Dict[str, float] = field(default_factory=dict)
    mAPH_by_mode: Dict[str, float] = field(default_factory=dict)
    per_class_ap_by_mode: Dict[str, Dict[str, float]] = field(default_factory=dict)
    num_frames: int = 0
    detailed_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "mAP_by_mode": dict(self.mAP_by_mode),
            "mAPH_by_mode": dict(self.mAPH_by_mode),
            "per_class_ap_by_mode": {k: dict(v) for k, v in self.per_class_ap_by_mode.items()},
            "num_frames": self.num_frames,
            "detailed_metrics": dict(self.detailed_metrics),
        }


@dataclass(frozen=True)
class _APEntry:
    """One per-class AP (or APH) value at a single matching threshold."""

    label: str
    mode: str
    threshold: Any
    value: float


@dataclass(frozen=True)
class _ModeAggregate:
    """Mean AP/APH across all classes for one matching mode."""

    mode: str
    map: Optional[float]
    maph: Optional[float]


class DetectionMetricsInterface(BaseMetricsInterface):
    """Shared logic for 2D/3D detection metrics interfaces."""

    # 3D detection has heading; 2D does not. Controls whether APH/mAPH keys are emitted.
    _supports_aph: bool = False

    def _extract_scores(
        self, metrics_score: MetricsScore
    ) -> Tuple[List[_APEntry], List[_APEntry], List[_ModeAggregate]]:
        """Read a ``MetricsScore`` into structured records.

        Returns:
            (ap_entries, aph_entries, aggregates). ``aph_entries`` is always populated when
            the score exposes heading data; callers gate emission on ``_supports_aph``.
        """
        ap_entries: List[_APEntry] = []
        aph_entries: List[_APEntry] = []
        aggregates: List[_ModeAggregate] = []

        for map_instance in metrics_score.mean_ap_values:
            mode = map_instance.matching_mode.value.lower().replace(" ", "_")

            for label, aps in map_instance.label_to_aps.items():
                label_name = label.value
                for ap in aps:
                    ap_entries.append(
                        _APEntry(label=label_name, mode=mode, threshold=ap.matching_threshold, value=ap.ap)
                    )

            label_to_aphs = getattr(map_instance, "label_to_aphs", None)
            if label_to_aphs:
                for label, aphs in label_to_aphs.items():
                    label_name = label.value
                    for aph in aphs:
                        value = getattr(aph, "aph", None)
                        if value is None:
                            value = getattr(aph, "ap", None)
                        if value is None:
                            continue
                        aph_entries.append(
                            _APEntry(label=label_name, mode=mode, threshold=aph.matching_threshold, value=value)
                        )

            aggregates.append(
                _ModeAggregate(mode=mode, map=map_instance.map, maph=getattr(map_instance, "maph", None))
            )

        return ap_entries, aph_entries, aggregates

    def _process_metrics_score(self, metrics_score: MetricsScore, prefix: Optional[str] = None) -> Dict[str, float]:
        """Flatten a ``MetricsScore`` into a flat metric dictionary.

        Key formats (``{p}`` is ``"{prefix}_"`` when a prefix is set, else empty):
            - ``{p}{label}_AP_{mode}_{threshold}``
            - ``{p}{label}_APH_{mode}_{threshold}`` (only when ``_supports_aph``)
            - ``{p}mAP_{mode}``
            - ``{p}mAPH_{mode}`` (only when ``_supports_aph``)
        """
        ap_entries, aph_entries, aggregates = self._extract_scores(metrics_score)
        key_prefix = f"{prefix}_" if prefix else ""

        metric_dict: Dict[str, float] = {}
        for e in ap_entries:
            metric_dict[f"{key_prefix}{e.label}_AP_{e.mode}_{e.threshold}"] = e.value
        if self._supports_aph:
            for e in aph_entries:
                metric_dict[f"{key_prefix}{e.label}_APH_{e.mode}_{e.threshold}"] = e.value
        for a in aggregates:
            metric_dict[f"{key_prefix}mAP_{a.mode}"] = a.map
            if self._supports_aph:
                metric_dict[f"{key_prefix}mAPH_{a.mode}"] = a.maph

        return metric_dict

    def _summarize_score(self, metrics_score: MetricsScore) -> DetectionSummary:
        """Build a :class:`DetectionSummary` from one evaluator's ``MetricsScore``.

        ``detailed_metrics`` carries the full (all-evaluator) metric dict from
        ``compute_metrics``; the per-mode aggregates come from this single score.
        """
        ap_entries, _aph_entries, aggregates = self._extract_scores(metrics_score)

        mAP_by_mode: Dict[str, float] = {a.mode: self._as_float(a.map) for a in aggregates}
        mAPH_by_mode: Dict[str, float] = (
            {a.mode: self._as_float(a.maph) for a in aggregates} if self._supports_aph else {}
        )

        # Per-class AP averaged across thresholds, grouped by matching mode.
        grouped: Dict[str, Dict[str, List[float]]] = {}
        for e in ap_entries:
            grouped.setdefault(e.mode, {}).setdefault(e.label, []).append(float(e.value))
        per_class_ap_by_mode: Dict[str, Dict[str, float]] = {
            mode: {label: float(np.mean(values)) for label, values in labels.items() if values}
            for mode, labels in grouped.items()
        }

        return DetectionSummary(
            mAP_by_mode=mAP_by_mode,
            mAPH_by_mode=mAPH_by_mode,
            per_class_ap_by_mode=per_class_ap_by_mode,
            num_frames=self._frame_count,
            detailed_metrics=self.compute_metrics(),
        )

    @staticmethod
    def _as_float(value: Optional[float]) -> float:
        """Coerce a possibly-None aggregate to a float, defaulting to 0.0."""
        return float(value) if value is not None else 0.0

    @property
    def summary(self) -> DetectionSummary:
        """Get a summary of the evaluation including mAP and per-class metrics per mode."""
        self.compute_metrics()
        metrics_score = self._select_summary_score()
        if metrics_score is None:
            return DetectionSummary(num_frames=self._frame_count, detailed_metrics=self._cached_metrics or {})
        return self._summarize_score(metrics_score)

    @abstractmethod
    def _select_summary_score(self) -> Optional[MetricsScore]:
        """Pick which evaluator's score the summary is built from (None if unavailable)."""
