"""Postprocessing for FRNet deployment.

Converts raw segmentation logits to class predictions with optional
score thresholding (below threshold -> ignore_index).
"""

from __future__ import annotations

from time import time

import numpy as np
import numpy.typing as npt
from mmengine.logging import MMLogger


class Postprocessing:
    """Argmax + threshold postprocessor."""

    def __init__(self, score_threshold: float = 0.0, ignore_index: int = 0) -> None:
        self._score_threshold = score_threshold
        self._ignore_index = ignore_index
        self.logger = MMLogger.get_current_instance()

    def postprocess(self, predictions: npt.NDArray[np.float32]) -> npt.NDArray[np.intp]:
        """Convert logits (N, num_classes) to per-point class indices (N,)."""
        t_start = time()
        result = np.where(
            np.max(predictions, axis=1) >= self._score_threshold,
            np.argmax(predictions, axis=1),
            self._ignore_index,
        )
        t_end = time()
        latency = np.round((t_end - t_start) * 1e3, 2)
        self.logger.info(f"Postprocessing latency: {latency} ms")
        return result
