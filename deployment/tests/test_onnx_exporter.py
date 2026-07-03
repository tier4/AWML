"""Unit tests for ONNXExporter's atomic, external-data-safe write path.

These exercise the staging-dir publish logic in ``_do_onnx_export`` without running a real
torch.onnx export: ``torch.onnx.export`` is monkeypatched to simulate what it writes (a main
``.onnx`` plus optional external-data sidecars), so the tests stay CPU-only and fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from deployment.export.exporters.configs import ONNXExportConfig
from deployment.export.exporters.onnx_exporter import ONNXExporter


def _exporter() -> ONNXExporter:
    return ONNXExporter(ONNXExportConfig())


def _no_staging_left(target: Path) -> bool:
    """No leftover staging directory next to the target after publish/cleanup."""
    return not (target.parent / f".{target.name}.staging").exists()


class TestAtomicExport:
    def test_successful_export_moves_into_place(self, tmp_path, monkeypatch):
        target = tmp_path / "out" / "model.onnx"
        cfg = ONNXExportConfig()

        def fake_export(model, args, f, **kwargs):
            Path(f).write_bytes(b"onnx-bytes")

        monkeypatch.setattr(torch.onnx, "export", fake_export)
        _exporter()._do_onnx_export(model=object(), sample_input=object(), output_path=str(target), export_cfg=cfg)

        assert target.read_bytes() == b"onnx-bytes"
        assert _no_staging_left(target)

    def test_external_data_sidecar_published_too(self, tmp_path, monkeypatch):
        target = tmp_path / "model.onnx"
        cfg = ONNXExportConfig()

        def fake_export(model, args, f, **kwargs):
            # Simulate a >2GB export: a main file plus an external-data sidecar next to it.
            Path(f).write_bytes(b"graph")
            Path(f).with_name(Path(f).name + ".data").write_bytes(b"weights")

        monkeypatch.setattr(torch.onnx, "export", fake_export)
        _exporter()._do_onnx_export(model=object(), sample_input=object(), output_path=str(target), export_cfg=cfg)

        assert target.read_bytes() == b"graph"
        assert (tmp_path / "model.onnx.data").read_bytes() == b"weights"
        assert _no_staging_left(target)

    def test_failed_export_leaves_no_partial_artifact(self, tmp_path, monkeypatch):
        target = tmp_path / "model.onnx"
        cfg = ONNXExportConfig()

        def fake_export(model, args, f, **kwargs):
            Path(f).write_bytes(b"partial")  # something is written...
            raise ValueError("boom")  # ...then export fails

        monkeypatch.setattr(torch.onnx, "export", fake_export)
        with pytest.raises(RuntimeError, match="ONNX export failed"):
            _exporter()._do_onnx_export(model=object(), sample_input=object(), output_path=str(target), export_cfg=cfg)

        assert not target.exists()
        assert _no_staging_left(target)

    def test_failed_export_preserves_previous_good_model(self, tmp_path, monkeypatch):
        target = tmp_path / "model.onnx"
        target.write_bytes(b"previous-good")
        cfg = ONNXExportConfig()

        def fake_export(model, args, f, **kwargs):
            raise ValueError("boom")

        monkeypatch.setattr(torch.onnx, "export", fake_export)
        with pytest.raises(RuntimeError):
            _exporter()._do_onnx_export(model=object(), sample_input=object(), output_path=str(target), export_cfg=cfg)

        # The valid pre-existing artifact must survive a failed re-export.
        assert target.read_bytes() == b"previous-good"
        assert _no_staging_left(target)
