"""
Verification orchestration for deployment workflows.

This module handles scenario-based verification across different backends.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from deployment.configs import BaseDeploymentConfig
from deployment.core.backend import Backend
from deployment.core.evaluation.base_evaluator import BaseEvaluator
from deployment.core.evaluation.evaluator_types import ModelSpec
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.runtime.artifact_manager import ArtifactManager


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
        evaluator: BaseEvaluator,
        data_loader: BaseDataLoader,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize verification orchestrator.

        Args:
            config: Deployment configuration
            evaluator: Evaluator instance for running verification
            data_loader: Data loader for loading samples
            logger: Logger instance
        """
        self.config = config
        self.evaluator = evaluator
        self.data_loader = data_loader
        self.logger = logger

    def run(self, artifact_manager: ArtifactManager) -> Dict[str, Any]:
        """
        Run verification on exported models using policy-based verification.

        Args:
            artifact_manager: Artifact manager for resolving model paths
        Returns:
            Verification results dictionary
        """
        verification_cfg = self.config.verification_config

        if not verification_cfg.enabled:
            self.logger.info("Verification disabled (verification.enabled=False), skipping...")
            return {}

        export_mode = self.config.export_config.mode
        scenarios = self.config.get_verification_scenarios(export_mode)

        if not scenarios:
            self.logger.info(f"No verification scenarios for export mode '{export_mode.value}', skipping...")
            return {}

        _, pytorch_valid = artifact_manager.resolve_artifact(Backend.PYTORCH)
        if not pytorch_valid:
            self.logger.warning(
                "PyTorch checkpoint not registered or missing; verification needs it for preprocessing/decode. "
                "Skipping verification."
            )
            return {}

        num_verify_samples = verification_cfg.num_verify_samples
        tolerance = verification_cfg.tolerance
        self.logger.info("=" * 80)
        self.logger.info(f"Running Verification (mode: {export_mode.value})")
        self.logger.info("=" * 80)

        all_results: Dict[str, Any] = {}
        total_passed = 0
        total_failed = 0

        for i, policy in enumerate(scenarios):
            ref_device = policy.ref_device
            test_device = policy.test_device

            self.logger.info(
                f"\nScenario {i+1}/{len(scenarios)}: "
                f"{policy.ref_backend.value}({ref_device}) vs {policy.test_backend.value}({test_device})"
            )

            ref_artifact, ref_valid = artifact_manager.resolve_artifact(policy.ref_backend)
            test_artifact, test_valid = artifact_manager.resolve_artifact(policy.test_backend)

            if not ref_valid or not test_valid:
                ref_path = ref_artifact.path if ref_artifact else None
                test_path = test_artifact.path if test_artifact else None
                self.logger.warning(
                    "  Skipping: missing or invalid artifacts "
                    f"(ref={ref_path}, valid={ref_valid}, test={test_path}, valid={test_valid})"
                )
                continue

            reference_spec = ModelSpec(backend=policy.ref_backend, device=ref_device, artifact=ref_artifact)
            test_spec = ModelSpec(backend=policy.test_backend, device=test_device, artifact=test_artifact)

            verification_results = self.evaluator.verify(
                reference=reference_spec,
                test=test_spec,
                data_loader=self.data_loader,
                num_samples=num_verify_samples,
                tolerance=tolerance,
                verbose=False,
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
                    self.logger.info(f"Scenario {i+1} passed ({passed} comparisons)")
                else:
                    self.logger.warning(f"Scenario {i+1} failed ({failed}/{passed+failed} comparisons)")

        self.logger.info("\n" + "=" * 80)
        if total_failed == 0:
            self.logger.info(f"All verifications passed! ({total_passed} total)")
        else:
            self.logger.warning(f"{total_failed}/{total_passed + total_failed} verifications failed")
        self.logger.info("=" * 80)

        all_results["summary"] = {
            "passed": total_passed,
            "failed": total_failed,
            "total": total_passed + total_failed,
        }

        return all_results
