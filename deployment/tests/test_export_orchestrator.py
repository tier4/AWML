"""Unit tests for ExportOrchestrator control flow.

Covers the two behaviors this stack relies on for correctness:
  * the stale-ONNX guard (a requested ONNX export that produces nothing must abort the run,
    never fall through to TensorRT with a stale ONNX);
  * external-artifact resolution delegating to ArtifactManager (single source of truth).

These are pure control-flow tests: model loading and the actual export steps are stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deployment.config.enums import Backend
from deployment.primitives.artifacts import Artifact
from deployment.runtime.export_orchestrator import ExportOrchestrator, ExportResult


def _orchestrator(export_config, artifact_manager=None) -> ExportOrchestrator:
    config = SimpleNamespace(checkpoint_path="unused.pth", export_config=export_config)
    return ExportOrchestrator(
        config=config,
        data_loader=Mock(),
        artifact_manager=artifact_manager or Mock(),
        model_loader=Mock(),
    )


class TestStaleOnnxGuard:
    def test_requested_onnx_producing_nothing_aborts_run(self):
        export_config = SimpleNamespace(should_export_onnx=True, should_export_tensorrt=True, onnx_path="stale/onnx")
        orch = _orchestrator(export_config)
        # Bypass real model loading and force ONNX export to produce nothing.
        orch._load_and_register_pytorch_model = lambda ckpt: object()
        orch._run_onnx_export = lambda model: None

        with pytest.raises(RuntimeError, match="stale"):
            orch.run()

    def test_does_not_reach_tensorrt_when_onnx_fails(self):
        export_config = SimpleNamespace(should_export_onnx=True, should_export_tensorrt=True, onnx_path="stale/onnx")
        orch = _orchestrator(export_config)
        orch._load_and_register_pytorch_model = lambda ckpt: object()
        orch._run_onnx_export = lambda model: None
        orch._run_tensorrt_export = Mock(side_effect=AssertionError("TensorRT must not run on stale ONNX"))

        with pytest.raises(RuntimeError):
            orch.run()
        orch._run_tensorrt_export.assert_not_called()


class TestExternalArtifactResolution:
    def test_resolution_delegates_to_artifact_manager(self):
        manager = Mock()

        def resolve(backend):
            if backend == Backend.ONNX:
                return Artifact(path="/models/model.onnx"), True
            return None, False

        manager.resolve_artifact.side_effect = resolve
        orch = _orchestrator(SimpleNamespace(), artifact_manager=manager)

        result = ExportResult()
        orch._resolve_external_artifacts(result)

        assert result.onnx_path == "/models/model.onnx"
        assert result.tensorrt_path is None

    def test_configured_path_that_does_not_exist_is_ignored(self):
        manager = Mock()
        manager.resolve_artifact.return_value = (Artifact(path="/missing/model.engine"), False)
        orch = _orchestrator(SimpleNamespace(), artifact_manager=manager)

        result = ExportResult()
        orch._resolve_external_artifacts(result)

        assert result.onnx_path is None
        assert result.tensorrt_path is None
