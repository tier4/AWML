"""
Base Deployment Pipeline for Unified Model Deployment.

Flattened from `deployment/pipelines/common/base_pipeline.py`.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch

from deployment.core.evaluation.evaluator_types import InferenceResult

logger = logging.getLogger(__name__)


class BaseDeploymentPipeline(ABC):
    def __init__(self, model: Any, device: str = "cpu", task_type: str = "unknown", backend_type: str = "unknown"):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.task_type = task_type
        self.backend_type = backend_type
        self._stage_latencies: Dict[str, float] = {}

        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")

    @abstractmethod
    def preprocess(self, input_data: Any, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run_model(self, preprocessed_input: Any) -> Union[Any, Tuple[Any, Dict[str, float]]]:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, model_output: Any, metadata: Dict = None) -> Any:
        raise NotImplementedError

    def infer(
        self, input_data: Any, metadata: Optional[Dict] = None, return_raw_outputs: bool = False, **kwargs
    ) -> InferenceResult:
        if metadata is None:
            metadata = {}

        latency_breakdown: Dict[str, float] = {}

        try:
            start_time = time.perf_counter()

            preprocessed = self.preprocess(input_data, **kwargs)

            preprocess_metadata = {}
            model_input = preprocessed
            if isinstance(preprocessed, tuple) and len(preprocessed) == 2 and isinstance(preprocessed[1], dict):
                model_input, preprocess_metadata = preprocessed

            preprocess_time = time.perf_counter()
            latency_breakdown["preprocessing_ms"] = (preprocess_time - start_time) * 1000

            merged_metadata = {}
            merged_metadata.update(metadata or {})
            merged_metadata.update(preprocess_metadata)

            model_start = time.perf_counter()
            model_result = self.run_model(model_input)
            model_time = time.perf_counter()
            latency_breakdown["model_ms"] = (model_time - model_start) * 1000

            if isinstance(model_result, tuple) and len(model_result) == 2:
                model_output, stage_latencies = model_result
                if isinstance(stage_latencies, dict):
                    latency_breakdown.update(stage_latencies)
            else:
                model_output = model_result

            # Legacy stage latency aggregation (kept)
            if hasattr(self, "_stage_latencies") and isinstance(self._stage_latencies, dict):
                latency_breakdown.update(self._stage_latencies)
                self._stage_latencies = {}

            total_latency = (time.perf_counter() - start_time) * 1000

            if return_raw_outputs:
                return InferenceResult(output=model_output, latency_ms=total_latency, breakdown=latency_breakdown)

            postprocess_start = time.perf_counter()
            predictions = self.postprocess(model_output, merged_metadata)
            postprocess_time = time.perf_counter()
            latency_breakdown["postprocessing_ms"] = (postprocess_time - postprocess_start) * 1000

            total_latency = (time.perf_counter() - start_time) * 1000
            return InferenceResult(output=predictions, latency_ms=total_latency, breakdown=latency_breakdown)

        except Exception:
            logger.exception("Inference failed.")
            raise

    def cleanup(self) -> None:
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"task={self.task_type}, "
            f"backend={self.backend_type})"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
