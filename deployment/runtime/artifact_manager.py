"""
Artifact management for deployment workflows.

This module handles registration and resolution of model artifacts (PyTorch checkpoints,
ONNX models, TensorRT engines) across different backends.
"""

import logging
from typing import Dict, Optional, Tuple

from deployment.config.base import BaseDeploymentConfig
from deployment.config.enums import Backend
from deployment.primitives.artifacts import Artifact

logger = logging.getLogger(__name__)


class ArtifactManager:
    """
    Manages model artifacts and path resolution for deployment workflows.

    Resolution Order (consistent for all backends):
    1. Registered artifacts (from export operations) - highest priority
    2. Explicit paths from evaluation.backends.<backend> config:
       - ONNX: evaluation.backends.onnx.model_dir
       - TensorRT: evaluation.backends.tensorrt.engine_dir
    3. Backend-specific fallback paths:
       - PyTorch: checkpoint_path
       - ONNX: export.onnx_path
    """

    def __init__(self, config: BaseDeploymentConfig) -> None:
        """
        Initialize artifact manager.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.artifacts: Dict[str, Artifact] = {}

    def register_artifact(self, backend: Backend, artifact: Artifact) -> None:
        """
        Register an artifact for a given backend.

        Args:
            backend: Backend to register the artifact for
            artifact: Artifact to register
        """
        self.artifacts[backend.value] = artifact
        logger.debug("Registered %s artifact: %s", backend.value, artifact.path)

    def resolve_artifact(self, backend: Backend) -> Tuple[Optional[Artifact], bool]:
        """
        Resolve an artifact for a given backend.

        Args:
            backend: Backend to resolve the artifact for
        Returns:
            Tuple containing the artifact and a boolean indicating if the artifact exists
        """
        artifact = self.artifacts.get(backend.value)
        if artifact:
            return artifact, artifact.exists

        config_path = self._get_config_path(backend)
        if config_path:
            artifact = Artifact(path=config_path)
            return artifact, artifact.exists

        return None, False

    def _get_config_path(self, backend: Backend) -> Optional[str]:
        """
        Get the configuration path for a given backend.

        Args:
            backend: Backend to get the configuration path for
        Returns:
            Configuration path for the given backend
        """
        eval_backends = self.config.evaluation_config.backends
        backend_cfg = eval_backends.get(backend.value) if eval_backends else None
        if backend_cfg is not None:
            if backend == Backend.ONNX and backend_cfg.model_dir:
                return backend_cfg.model_dir
            if backend == Backend.TENSORRT and backend_cfg.engine_dir:
                return backend_cfg.engine_dir

        if backend == Backend.PYTORCH:
            return self.config.checkpoint_path
        if backend == Backend.ONNX:
            return self.config.export_config.onnx_path

        return None
