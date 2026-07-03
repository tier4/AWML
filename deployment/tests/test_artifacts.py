"""Unit tests for artifact path resolution and ArtifactManager priority order."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deployment.config.enums import Backend
from deployment.config.schema import BackendEvalConfig
from deployment.primitives.artifacts import Artifact, resolve_artifact_path
from deployment.runtime.artifact_manager import ArtifactManager


# --------------------------------------------------------------------------------------
# resolve_artifact_path (pure path logic)
# --------------------------------------------------------------------------------------
class TestResolveArtifactPath:
    def test_resolves_existing_file(self, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"x")
        out = resolve_artifact_path(
            base_dir=str(tmp_path),
            components_cfg={"model": {"onnx_file": "model.onnx"}},
            component_name="model",
            file_key="onnx_file",
        )
        assert out == str((tmp_path / "model.onnx").resolve())

    def test_base_dir_must_be_directory(self, tmp_path):
        f = tmp_path / "not_a_dir"
        f.write_bytes(b"x")
        with pytest.raises(ValueError):
            resolve_artifact_path(
                base_dir=str(f),
                components_cfg={"model": {"onnx_file": "model.onnx"}},
                component_name="model",
                file_key="onnx_file",
            )

    def test_missing_filename_raises_keyerror(self, tmp_path):
        with pytest.raises(KeyError):
            resolve_artifact_path(
                base_dir=str(tmp_path),
                components_cfg={"model": {}},
                component_name="model",
                file_key="onnx_file",
            )

    def test_absolute_filename_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            resolve_artifact_path(
                base_dir=str(tmp_path),
                components_cfg={"model": {"onnx_file": "/abs/model.onnx"}},
                component_name="model",
                file_key="onnx_file",
            )

    def test_escaping_base_dir_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            resolve_artifact_path(
                base_dir=str(tmp_path),
                components_cfg={"model": {"onnx_file": "../escape.onnx"}},
                component_name="model",
                file_key="onnx_file",
            )

    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_artifact_path(
                base_dir=str(tmp_path),
                components_cfg={"model": {"onnx_file": "absent.onnx"}},
                component_name="model",
                file_key="onnx_file",
            )


# --------------------------------------------------------------------------------------
# ArtifactManager resolution priority
# --------------------------------------------------------------------------------------
def _stub_config(tmp_path):
    return SimpleNamespace(
        checkpoint_path=str(tmp_path / "ckpt.pth"),
        export_config=SimpleNamespace(onnx_path=str(tmp_path / "onnx_export")),
        evaluation_config=SimpleNamespace(
            backends={
                "onnx": BackendEvalConfig(model_dir=str(tmp_path / "cfg_onnx")),
                "tensorrt": BackendEvalConfig(engine_dir=str(tmp_path / "cfg_trt")),
            }
        ),
    )


class TestArtifactManager:
    def test_registered_artifact_takes_priority(self, tmp_path):
        ckpt = tmp_path / "real_ckpt.pth"
        ckpt.write_bytes(b"x")
        mgr = ArtifactManager(_stub_config(tmp_path))
        mgr.register_artifact(Backend.PYTORCH, Artifact(path=str(ckpt)))

        artifact, exists = mgr.resolve_artifact(Backend.PYTORCH)
        assert artifact is not None
        assert artifact.path == str(ckpt)
        assert exists is True

    def test_falls_back_to_eval_backend_config(self, tmp_path):
        mgr = ArtifactManager(_stub_config(tmp_path))
        artifact, exists = mgr.resolve_artifact(Backend.TENSORRT)
        assert artifact is not None
        assert artifact.path == str(tmp_path / "cfg_trt")
        # Path does not actually exist -> exists is False (no silent success).
        assert exists is False

    def test_pytorch_falls_back_to_checkpoint_path(self, tmp_path):
        mgr = ArtifactManager(_stub_config(tmp_path))
        artifact, _ = mgr.resolve_artifact(Backend.PYTORCH)
        assert artifact is not None
        assert artifact.path == str(tmp_path / "ckpt.pth")
