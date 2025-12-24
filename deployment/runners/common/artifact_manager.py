"""
Artifact management for deployment workflows.

This module handles registration and resolution of model artifacts (PyTorch checkpoints,
ONNX models, TensorRT engines) across different backends.
"""

import logging
import os.path as osp
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

from deployment.core.artifacts import Artifact
from deployment.core.backend import Backend
from deployment.core.config.base_config import BaseDeploymentConfig


class ArtifactManager:
    """
    Manages model artifacts and path resolution for deployment workflows.

    This class centralizes all logic for:
    - Registering artifacts after export
    - Resolving artifact paths from configuration
    - Validating artifact existence
    - Looking up artifacts by backend

    Resolution Order (consistent for all backends):
    1. Registered artifacts (from export operations) - highest priority
    2. Explicit paths from evaluation.backends.<backend> config:
       - ONNX: evaluation.backends.onnx.model_dir
       - TensorRT: evaluation.backends.tensorrt.engine_dir
    3. Backend-specific fallback paths:
       - PyTorch: checkpoint_path
       - ONNX: export.onnx_path
    """

    def __init__(self, config: BaseDeploymentConfig, logger: logging.Logger):
        """
        Initialize artifact manager.

        Args:
            config: Deployment configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.artifacts: Dict[str, Artifact] = {}

    def register_artifact(self, backend: Backend, artifact: Artifact) -> None:
        """
        Register an artifact for a backend.

        Args:
            backend: Backend identifier
            artifact: Artifact to register
        """
        self.artifacts[backend.value] = artifact
        self.logger.debug(f"Registered {backend.value} artifact: {artifact.path}")

    def get_artifact(self, backend: Backend) -> Optional[Artifact]:
        """
        Get registered artifact for a backend.

        Args:
            backend: Backend identifier

        Returns:
            Artifact if found, None otherwise
        """
        return self.artifacts.get(backend.value)

    def resolve_artifact(self, backend: Backend) -> Tuple[Optional[Artifact], bool]:
        """
        Resolve artifact for any backend with consistent resolution order.

        Resolution order (same for all backends):
        1. Registered artifact (from previous export/load operations)
        2. Explicit path from evaluation.backends.<backend> config:
           - ONNX: model_dir
           - TensorRT: engine_dir
        3. Backend-specific fallback (checkpoint_path for PyTorch, export.onnx_path for ONNX)

        Args:
            backend: Backend identifier

        Returns:
            Tuple of (artifact, is_valid).
            artifact is an Artifact instance if a path could be resolved, otherwise None.
            is_valid indicates whether the artifact exists on disk.
        """
        # Priority 1: Check registered artifacts
        artifact = self.artifacts.get(backend.value)
        if artifact:
            return artifact, artifact.exists()

        # Priority 2 & 3: Get path from config
        config_path = self._get_config_path(backend)
        if config_path:
            is_dir = osp.isdir(config_path) if osp.exists(config_path) else False
            artifact = Artifact(path=config_path, multi_file=is_dir)
            return artifact, artifact.exists()

        return None, False

    def _get_config_path(self, backend: Backend) -> Optional[str]:
        """
        Get artifact path from configuration.

        Resolution order:
        1. evaluation.backends.<backend>.model_dir or engine_dir (explicit per-backend path)
        2. Backend-specific fallbacks (checkpoint_path, export.onnx_path)

        Args:
            backend: Backend identifier

        Returns:
            Path string if found in config, None otherwise
        """
        # Priority 1: Check evaluation.backends.<backend> for explicit path
        eval_backends = self.config.evaluation_config.backends
        backend_cfg = self._get_backend_entry(eval_backends, backend)
        if backend_cfg and isinstance(backend_cfg, Mapping):
            # ONNX uses model_dir, TensorRT uses engine_dir
            if backend == Backend.ONNX:
                path = backend_cfg.get("model_dir")
                if path:
                    return path
            elif backend == Backend.TENSORRT:
                path = backend_cfg.get("engine_dir")
                if path:
                    return path

        # Priority 2: Backend-specific fallbacks from export config
        if backend == Backend.PYTORCH:
            return self.config.checkpoint_path
        elif backend == Backend.ONNX:
            return self.config.export_config.onnx_path
        # TensorRT has no global fallback path in export config
        return None

    @staticmethod
    def _get_backend_entry(mapping: Optional[Mapping], backend: Backend) -> Any:
        """
        Fetch a config value that may be keyed by either string literals or Backend enums.

        Args:
            mapping: Configuration mapping (may be None or MappingProxyType)
            backend: Backend to look up

        Returns:
            Value from mapping if found, None otherwise
        """
        if not mapping:
            return None

        value = mapping.get(backend.value)
        if value is not None:
            return value

        return mapping.get(backend)
