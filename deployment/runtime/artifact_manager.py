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
        self.config = config
        self.logger = logger
        self.artifacts: Dict[str, Artifact] = {}

    def register_artifact(self, backend: Backend, artifact: Artifact) -> None:
        self.artifacts[backend.value] = artifact
        self.logger.debug(f"Registered {backend.value} artifact: {artifact.path}")

    def get_artifact(self, backend: Backend) -> Optional[Artifact]:
        return self.artifacts.get(backend.value)

    def resolve_artifact(self, backend: Backend) -> Tuple[Optional[Artifact], bool]:
        artifact = self.artifacts.get(backend.value)
        if artifact:
            return artifact, artifact.exists()

        config_path = self._get_config_path(backend)
        if config_path:
            is_dir = osp.isdir(config_path) if osp.exists(config_path) else False
            artifact = Artifact(path=config_path, multi_file=is_dir)
            return artifact, artifact.exists()

        return None, False

    def _get_config_path(self, backend: Backend) -> Optional[str]:
        eval_backends = self.config.evaluation_config.backends
        backend_cfg = self._get_backend_entry(eval_backends, backend)
        if backend_cfg and isinstance(backend_cfg, Mapping):
            if backend == Backend.ONNX:
                path = backend_cfg.get("model_dir")
                if path:
                    return path
            elif backend == Backend.TENSORRT:
                path = backend_cfg.get("engine_dir")
                if path:
                    return path

        if backend == Backend.PYTORCH:
            return self.config.checkpoint_path
        if backend == Backend.ONNX:
            return self.config.export_config.onnx_path

        return None

    @staticmethod
    def _get_backend_entry(mapping: Optional[Mapping], backend: Backend) -> Any:
        if not mapping:
            return None

        value = mapping.get(backend.value)
        if value is not None:
            return value

        return mapping.get(backend)
