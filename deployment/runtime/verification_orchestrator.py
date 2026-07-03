"""
Verification orchestration for deployment workflows.

This module handles scenario-based verification across different backends.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from deployment.config.base import BaseDeploymentConfig
from deployment.config.enums import Backend
from deployment.evaluation.backend_verifier import BackendVerifier
from deployment.evaluation.evaluator_types import ModelSpec
from deployment.io.base_data_loader import BaseDataLoader
from deployment.runtime.artifact_manager import ArtifactManager

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
        Run verification on exported models using policy-based verification.

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
        logger.info("=" * 80)
        logger.info("Running Verification (mode: %s)", export_mode.value)
        logger.info("=" * 80)

        all_results: Dict[str, Any] = {}
        total_passed = 0
        total_failed = 0

        for i, policy in enumerate(scenarios):
            ref_device = policy.ref_device
            test_device = policy.test_device

            logger.info(
                "\nScenario %s/%s: %s(%s) vs %s(%s)",
                i + 1,
                len(scenarios),
                policy.ref_backend.value,
                ref_device,
                policy.test_backend.value,
                test_device,
            )

            ref_artifact, ref_valid = self.artifact_manager.resolve_artifact(policy.ref_backend)
            test_artifact, test_valid = self.artifact_manager.resolve_artifact(policy.test_backend)

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

            reference_spec = ModelSpec(backend=policy.ref_backend, device=ref_device, artifact=ref_artifact)
            test_spec = ModelSpec(backend=policy.test_backend, device=test_device, artifact=test_artifact)

            verification_results = self.verifier.run(
                reference=reference_spec,
                test=test_spec,
                data_loader=self.data_loader,
                num_samples=num_verify_samples,
                tolerance=tolerance,
            )

            policy_key = f"{policy.ref_backend.value}_{ref_device}_vs_{policy.test_backend.value}_{test_device}"
            all_results[policy_key] = verification_results

            if "summary" in verification_results:
                summary = verification_results["summary"]
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                total_passed += passed
                total_failed += failed
                if failed == 0:
                    logger.info("Scenario %s passed (%s comparisons)", i + 1, passed)
                else:
                    logger.warning(
                        "Scenario %s failed (%s/%s comparisons)",
                        i + 1,
                        failed,
                        passed + failed,
                    )

        logger.info("\n" + "=" * 80)
        if total_failed == 0:
            logger.info("All verifications passed! (%s total)", total_passed)
        else:
            logger.warning(
                "%s/%s verifications failed",
                total_failed,
                total_passed + total_failed,
            )
        logger.info("=" * 80)

        all_results["summary"] = {
            "passed": total_passed,
            "failed": total_failed,
            "total": total_passed + total_failed,
        }

        return all_results
