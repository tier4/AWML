"""
Evaluation orchestration for deployment workflows.

This module handles cross-backend evaluation with consistent metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

from deployment.core.backend import Backend
from deployment.core.config.base_config import BaseDeploymentConfig
from deployment.core.evaluation.base_evaluator import BaseEvaluator
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.runners.common.artifact_manager import ArtifactManager


class EvaluationOrchestrator:
    """
    Orchestrates evaluation across backends with consistent metrics.

    This class handles:
    - Resolving models to evaluate from configuration
    - Running evaluation for each enabled backend
    - Collecting and formatting evaluation results
    - Logging evaluation progress and results
    - Cross-backend metric comparison
    """

    def __init__(
        self,
        config: BaseDeploymentConfig,
        evaluator: BaseEvaluator,
        data_loader: BaseDataLoader,
        logger: logging.Logger,
    ):
        """
        Initialize evaluation orchestrator.

        Args:
            config: Deployment configuration
            evaluator: Evaluator instance for running evaluation
            data_loader: Data loader for loading samples
            logger: Logger instance
        """
        self.config = config
        self.evaluator = evaluator
        self.data_loader = data_loader
        self.logger = logger

    def run(self, artifact_manager: ArtifactManager) -> Dict[str, Any]:
        """
        Run evaluation on specified models.

        Args:
            artifact_manager: Artifact manager for resolving model paths

        Returns:
            Dictionary containing evaluation results for all backends
        """
        eval_config = self.config.evaluation_config

        if not eval_config.enabled:
            self.logger.info("Evaluation disabled, skipping...")
            return {}

        self.logger.info("=" * 80)
        self.logger.info("Running Evaluation")
        self.logger.info("=" * 80)

        # Get models to evaluate
        models_to_evaluate = self._get_models_to_evaluate(artifact_manager)

        if not models_to_evaluate:
            self.logger.warning("No models found for evaluation")
            return {}

        # Determine number of samples
        num_samples = eval_config.num_samples
        if num_samples == -1:
            num_samples = self.data_loader.get_num_samples()

        verbose_mode = eval_config.verbose

        # Run evaluation for each model
        all_results: Dict[str, Any] = {}

        for spec in models_to_evaluate:
            backend = spec.backend
            backend_device = self._normalize_device_for_backend(backend, spec.device)

            normalized_spec = ModelSpec(backend=backend, device=backend_device, artifact=spec.artifact)

            self.logger.info(f"\nEvaluating {backend.value} on {backend_device}...")

            try:
                results = self.evaluator.evaluate(
                    model=normalized_spec,
                    data_loader=self.data_loader,
                    num_samples=num_samples,
                    verbose=verbose_mode,
                )

                all_results[backend.value] = results

                self.logger.info(f"\n{backend.value.upper()} Results:")
                self.evaluator.print_results(results)

            except Exception as e:
                self.logger.error(f"Evaluation failed for {backend.value}: {e}", exc_info=True)
                all_results[backend.value] = {"error": str(e)}
            finally:
                # Ensure CUDA memory is cleaned up between model evaluations
                from deployment.pipelines.common.gpu_resource_mixin import clear_cuda_memory

                clear_cuda_memory()

        # Print cross-backend comparison if multiple backends
        if len(all_results) > 1:
            self._print_cross_backend_comparison(all_results)

        return all_results

    def _get_models_to_evaluate(self, artifact_manager: ArtifactManager) -> List[ModelSpec]:
        """
        Get list of models to evaluate from config.

        Args:
            artifact_manager: Artifact manager for resolving paths

        Returns:
            List of ModelSpec instances describing models to evaluate
        """
        backends = self.config.get_evaluation_backends()
        models_to_evaluate: List[ModelSpec] = []

        for backend_key, backend_cfg in backends.items():
            backend_enum = Backend.from_value(backend_key)
            if not backend_cfg.get("enabled", False):
                continue

            device = str(backend_cfg.get("device", "cpu") or "cpu")

            # Use artifact_manager to resolve artifact
            artifact, is_valid = artifact_manager.resolve_artifact(backend_enum)

            if is_valid and artifact:
                spec = ModelSpec(backend=backend_enum, device=device, artifact=artifact)
                models_to_evaluate.append(spec)
                self.logger.info(f"  - {backend_enum.value}: {artifact.path} (device: {device})")
            elif artifact is not None:
                self.logger.warning(f"  - {backend_enum.value}: {artifact.path} (not found or invalid, skipping)")

        return models_to_evaluate

    def _normalize_device_for_backend(self, backend: Backend, device: str) -> str:
        """
        Normalize device string for specific backend.

        Args:
            backend: Backend identifier
            device: Device string from config

        Returns:
            Normalized device string
        """
        normalized_device = str(device or self._get_default_device(backend) or "cpu")

        if backend in (Backend.PYTORCH, Backend.ONNX):
            if normalized_device not in ("cpu",) and not normalized_device.startswith("cuda"):
                self.logger.warning(
                    f"Unsupported device '{normalized_device}' for backend '{backend.value}'. " "Falling back to CPU."
                )
                normalized_device = "cpu"
        elif backend is Backend.TENSORRT:
            if not normalized_device or normalized_device == "cpu":
                normalized_device = self.config.devices.cuda or "cuda:0"
            if not normalized_device.startswith("cuda"):
                self.logger.warning(
                    "TensorRT evaluation requires CUDA device. "
                    f"Overriding device from '{normalized_device}' to 'cuda:0'."
                )
                normalized_device = "cuda:0"

        return normalized_device

    def _get_default_device(self, backend: Backend) -> str:
        """Return default device for a backend when config omits explicit value."""
        if backend is Backend.TENSORRT:
            return self.config.devices.cuda or "cuda:0"
        return self.config.devices.cpu or "cpu"

    def _print_cross_backend_comparison(self, all_results: Mapping[str, Any]) -> None:
        """
        Print cross-backend comparison of metrics.

        Args:
            all_results: Dictionary of results by backend
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Cross-Backend Comparison")
        self.logger.info("=" * 80)

        for backend_label, results in all_results.items():
            self.logger.info(f"\n{backend_label.upper()}:")
            if results and "error" not in results:
                # Print primary metrics
                if "accuracy" in results:
                    self.logger.info(f"  Accuracy: {results.get('accuracy', 0):.4f}")
                if "mAP" in results:
                    self.logger.info(f"  mAP: {results.get('mAP', 0):.4f}")

                # Print latency stats
                if "latency_stats" in results:
                    stats = results["latency_stats"]
                    self.logger.info(f"  Latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
                elif "latency" in results:
                    latency = results["latency"]
                    self.logger.info(f"  Latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
            else:
                self.logger.info("  No results available")
