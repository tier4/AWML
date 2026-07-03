"""
Export sample extractor: the seam that produces the tracing sample.

``SampleExtractor`` is the contract the ONNX export pipeline depends on to
obtain the per-sample payload used for tracing. ``DefaultSampleExtractor`` is the
built-in implementation for the common whole-model case (just preprocess the
loader's sample); projects that need model-specific feature extraction (e.g.
CenterPoint) provide their own extractor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from deployment.io.base_data_loader import BaseDataLoader


class SampleExtractor(ABC):
    """Interface for model-specific sample extraction for export.

    Implementations convert model-specific feature extraction outputs
    into a sample object that component builders can consume.
    """

    @abstractmethod
    def extract_sample(
        self,
        model: torch.nn.Module,
        data_loader: Any,
        sample_idx: int,
    ) -> Any:
        """Extract model-specific sample payload for export.

        Args:
            model: PyTorch model used for feature extraction
            data_loader: Data loader used to access the sample
            sample_idx: Sample index used for tracing/feature extraction

        Returns:
            Model-specific typed sample payload.
        """
        ...


class DefaultSampleExtractor(SampleExtractor):
    """Default sample extractor: returns the preprocessed model input for tracing.

    Used when a project does not supply its own extractor. No model-specific
    feature extraction is performed: the sample is simply the loader's
    preprocessed sample, used directly as the model's tracing input.
    """

    def extract_sample(
        self,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        sample_idx: int,
    ) -> Any:
        """Return the preprocessed input tensor(s) for the given sample.

        Args:
            model: PyTorch model (unused; the whole model is traced as-is).
            data_loader: Loader used to fetch and preprocess the sample.
            sample_idx: Index of the sample to load.

        Returns:
            Preprocessed input consumed directly by the ONNX exporter.
        """
        return data_loader.preprocess(data_loader.load_sample(sample_idx))
