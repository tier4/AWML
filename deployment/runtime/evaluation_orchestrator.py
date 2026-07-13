"""
Evaluation orchestration for deployment workflows.

This module handles cross-backend evaluation with consistent metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from deployment.config.base import BaseDeploymentConfig
from deployment.config.enums import Backend
from deployment.evaluation.base_evaluator import BaseEvaluator
from deployment.inference.gpu_resource_mixin import clear_cuda_memory
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.device import DeviceSpec
from deployment.primitives.evaluator_types import ModelSpec
from deployment.runtime.artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)


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
        artifact_manager: ArtifactManager,
    ):
        """
        Initialize the evaluation orchestrator.

        Args:
            config: Deployment configuration
            evaluator: Evaluator instance for running evaluation
            data_loader: Data loader for loading samples
            artifact_manager: Artifact manager for resolving model paths
        """
        self.config = config
        self.evaluator = evaluator
        self.data_loader = data_loader
        self.artifact_manager = artifact_manager

    def run(self) -> Dict[str, Any]:
        """
        Run the evaluation orchestration.

        Returns:
            Dictionary of evaluation results
        """
        eval_config = self.config.evaluation_config

        if not eval_config.enabled:
            logger.info("Evaluation disabled, skipping...")
            return {}

        logger.info("=" * 80)
        logger.info("Running Evaluation")
        logger.info("=" * 80)

        model_specs = self._resolve_model_specs()
        if not model_specs:
            logger.warning("No models found for evaluation")
            return {}

        num_samples = eval_config.num_samples
        if num_samples == -1:
            num_samples = self.data_loader.num_samples

        verbose = eval_config.verbose
        all_results: Dict[str, Any] = {}

        for model_spec in model_specs:
            backend = model_spec.backend
            logger.info("\nEvaluating %s on %s...", backend.value, model_spec.device)
            try:
                results = self.evaluator.evaluate(
                    model=model_spec,
                    data_loader=self.data_loader,
                    num_samples=num_samples,
                    verbose=verbose,
                    num_warmup=eval_config.num_warmup,
                )
                all_results[backend.value] = results
                logger.info("\n%s Results:", backend.value.upper())
                self.evaluator.print_results(results)
            except Exception as e:
                logger.error("Evaluation failed for %s: %s", backend.value, e, exc_info=True)
                all_results[backend.value] = {"error": str(e)}
            finally:
                clear_cuda_memory()

        if len(all_results) > 1:
            self._print_cross_backend_comparison(all_results)

        return all_results

    def _resolve_model_specs(self) -> List[ModelSpec]:
        """
        Resolve the model specs to evaluate from the configuration.

        For each enabled backend, resolves its device and artifact, keeping only
        backends whose artifact exists on disk.

        Returns:
            List of model specifications
        """
        backend_configs = self.config.evaluation_config.backends
        model_specs: List[ModelSpec] = []

        for backend_key, backend_cfg in backend_configs.items():
            backend_enum = Backend.from_value(backend_key)
            if not backend_cfg.enabled:
                continue

            device = self._resolve_device_for_backend(backend_enum, backend_cfg.device)
            artifact, artifact_exists = self.artifact_manager.resolve_artifact(backend_enum)

            if artifact_exists and artifact:
                model_specs.append(ModelSpec(backend=backend_enum, device=device, artifact=artifact))
                logger.info("  - %s: %s (device: %s)", backend_enum.value, artifact.path, device)
            elif artifact is not None:
                logger.warning(
                    "  - %s: %s (not found or invalid, skipping)",
                    backend_enum.value,
                    artifact.path,
                )

        return model_specs

    def _resolve_device_for_backend(self, backend: Backend, configured_device: Optional[Any]) -> DeviceSpec:
        """
        Resolve the single device a backend will run on (called once per backend).

        Falls back to the backend default when nothing is configured, and enforces
        backend constraints: a CUDA-only backend handed a non-CUDA device is overridden
        (with a warning) to the default CUDA device.

        Args:
            backend: Backend the device is being resolved for
            configured_device: Raw device from config (e.g. "cuda:0"), or None/blank to
                use the backend default
        Returns:
            The device the backend will actually use
        """
        resolved_device = (
            DeviceSpec.from_value(configured_device) if configured_device else self._get_default_device(backend)
        )

        if backend.requires_cuda and not resolved_device.is_cuda:
            default_device = self._get_default_device(backend)
            logger.warning(
                "%s evaluation requires CUDA device. Overriding device from '%s' to '%s'.",
                backend.value,
                resolved_device,
                default_device,
            )
            resolved_device = default_device

        return resolved_device

    def _get_default_device(self, backend: Backend) -> DeviceSpec:
        """
        Get the default device for a backend.

        Args:
            backend: Backend to get the default device for
        Returns:
            Default device (DeviceSpec) for the backend
        """
        if backend is Backend.TENSORRT:
            if self.config.device_config.cuda is None:
                raise RuntimeError("TensorRT backend requires a configured CUDA device.")
            return self.config.device_config.cuda
        return self.config.device_config.cpu

    def _print_cross_backend_comparison(self, all_results: Mapping[str, Any]) -> None:
        """
        Print the cross-backend comparison results.

        Args:
            all_results: Dictionary of all results
        """
        logger.info("\n" + "=" * 80)
        logger.info("Cross-Backend Comparison")
        logger.info("=" * 80)

        for backend_label, results in all_results.items():
            logger.info("\n%s:", backend_label.upper())
            if results and "error" not in results:
                for line in self.evaluator.summarize_for_comparison(results):
                    logger.info(line)
            else:
                logger.info("  No results available")
