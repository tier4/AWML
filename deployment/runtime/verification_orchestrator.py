"""
Verification orchestration for deployment workflows.

This module handles scenario-based verification across different backends.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from deployment.config.base import BaseDeploymentConfig
from deployment.config.enums import Backend
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.evaluator_types import ModelSpec
from deployment.runtime.artifact_manager import ArtifactManager
from deployment.verification.backend_verifier import BackendVerifier
from deployment.verification.reporting import banner

logger = logging.getLogger(__name__)


class VerificationOrchestrator:
    """
    Orchestrates verification across backends using scenario-based verification.

    This class handles:
    - Running verification scenarios from config
    - Resolving model paths via ArtifactManager
    - Collecting and aggregating verification results
    - Logging verification progress and results
    """

    def __init__(
        self,
        config: BaseDeploymentConfig,
        verifier: BackendVerifier,
        data_loader: BaseDataLoader,
        artifact_manager: ArtifactManager,
    ) -> None:
        """
        Initialize verification orchestrator.

        Args:
            config: Deployment configuration
            verifier: Backend verifier that runs reference-vs-test comparisons
            data_loader: Data loader for loading samples
            artifact_manager: Artifact manager for resolving model paths
        """
        self.config = config
        self.verifier = verifier
        self.data_loader = data_loader
        self.artifact_manager = artifact_manager

    def run(self) -> Dict[str, Any]:
        """
        Run verification on exported models using scenario-based verification.

        Returns:
            Verification results dictionary
        """
        verification_cfg = self.config.verification_config

        if not verification_cfg.enabled:
            logger.info("Verification disabled (verification.enabled=False), skipping...")
            return {}

        export_mode = self.config.export_config.mode
        scenarios = self.config.get_verification_scenarios(export_mode)

        if not scenarios:
            logger.info(
                "No verification scenarios for export mode '%s', skipping...",
                export_mode.value,
            )
            return {}

        _, pytorch_valid = self.artifact_manager.resolve_artifact(Backend.PYTORCH)
        if not pytorch_valid:
            logger.warning(
                "PyTorch checkpoint not registered or missing; verification needs it for preprocessing/decode. "
                "Skipping verification."
            )
            return {}

        num_verify_samples = verification_cfg.num_verify_samples
        tolerance = verification_cfg.tolerance
        logger.info(banner())
        logger.info("Running Verification (mode: %s)", export_mode.value)
        logger.info(banner())

        all_results: Dict[str, Any] = {}
        total_passed = 0
        total_failed = 0

        for i, scenario in enumerate(scenarios):
            ref_device = scenario.ref_device
            test_device = scenario.test_device

            logger.info(
                "\nScenario %s/%s: %s(%s) vs %s(%s)",
                i + 1,
                len(scenarios),
                scenario.ref_backend.value,
                ref_device,
                scenario.test_backend.value,
                test_device,
            )

            ref_artifact, ref_valid = self.artifact_manager.resolve_artifact(scenario.ref_backend)
            test_artifact, test_valid = self.artifact_manager.resolve_artifact(scenario.test_backend)

            if not ref_valid or not test_valid:
                ref_path = ref_artifact.path if ref_artifact else None
                test_path = test_artifact.path if test_artifact else None
                logger.warning(
                    "  Skipping: missing or invalid artifacts (ref=%s, valid=%s, test=%s, valid=%s)",
                    ref_path,
                    ref_valid,
                    test_path,
                    test_valid,
                )
                continue

            reference_spec = ModelSpec(backend=scenario.ref_backend, device=ref_device, artifact=ref_artifact)
            test_spec = ModelSpec(backend=scenario.test_backend, device=test_device, artifact=test_artifact)

            verification_results = self.verifier.run(
                reference=reference_spec,
                test=test_spec,
                data_loader=self.data_loader,
                num_samples=num_verify_samples,
                tolerance=tolerance,
            )

            scenario_key = f"{scenario.ref_backend.value}_{ref_device}_vs_{scenario.test_backend.value}_{test_device}"
            all_results[scenario_key] = verification_results

            # Surface a pre-inference failure (e.g. device validation) instead of silently
            # counting the scenario as 0/0; BackendVerifier sets ``error`` in that case.
            if "error" in verification_results:
                logger.warning("Scenario %s could not run: %s", i + 1, verification_results["error"])
                continue

            if "summary" in verification_results:
                summary = verification_results["summary"]
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                total_passed += passed
                total_failed += failed
                if failed == 0:
                    logger.info("Scenario %s passed (%s samples)", i + 1, passed)
                else:
                    logger.warning(
                        "Scenario %s failed (%s/%s samples)",
                        i + 1,
                        failed,
                        passed + failed,
                    )

        logger.info("\n" + banner())
        if total_failed == 0:
            logger.info("All verification samples passed! (%s total)", total_passed)
        else:
            logger.warning(
                "%s/%s verification samples failed",
                total_failed,
                total_passed + total_failed,
            )
        logger.info(banner())

        all_results["summary"] = {
            "passed": total_passed,
            "failed": total_failed,
            "total": total_passed + total_failed,
        }

        return all_results
