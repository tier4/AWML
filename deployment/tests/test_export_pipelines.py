"""Unit tests for the unified export pipelines.

Covers the per-component export control flow of ``OnnxExportPipeline``, the
whole-model default seam (``DefaultSampleExtractor`` / ``DefaultComponentBuilder``),
and the input validation of ``TensorRTExportPipeline``. The actual exporters and
data loading are stubbed; these are pure control-flow / path-construction tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from deployment.export.pipelines.component_builder import DefaultComponentBuilder, ExportableComponent
from deployment.export.pipelines.onnx_export_pipeline import OnnxExportPipeline
from deployment.export.pipelines.sample_extractor import DefaultSampleExtractor
from deployment.export.pipelines.tensorrt_export_pipeline import TensorRTExportPipeline
from deployment.primitives.artifacts import Artifact


def _onnx_config(component_settings: dict) -> SimpleNamespace:
    """Build a fake deploy config exposing the bits the ONNX pipeline reads."""
    return SimpleNamespace(get_onnx_settings=lambda name: component_settings[name])


def _builder(components: list[ExportableComponent]) -> Mock:
    """A component builder stub that returns the given components."""
    builder = Mock()
    builder.build_components.return_value = components
    return builder


class TestOnnxExportPipeline:
    def test_exports_one_file_per_component(self, tmp_path):
        exporter = Mock()

        voxel_module, head_module = object(), object()
        components = [
            ExportableComponent(name="voxel", module=voxel_module, sample_input="VOXEL_IN"),
            ExportableComponent(name="head", module=head_module, sample_input="HEAD_IN"),
        ]
        extractor = Mock()
        extractor.extract_sample.return_value = "SAMPLE"

        pipeline = OnnxExportPipeline(
            sample_extractor=extractor,
            component_builder=_builder(components),
            onnx_wrapper_cls=Mock(),
        )

        config = _onnx_config(
            {
                "voxel": SimpleNamespace(save_file="voxel.onnx", batch_size=None),
                "head": SimpleNamespace(save_file="head.onnx", batch_size=None),
            }
        )
        model = object()
        data_loader = Mock()

        with patch(
            "deployment.export.pipelines.onnx_export_pipeline.ONNXExporter",
            return_value=exporter,
        ):
            artifact = pipeline.export(
                model=model,
                data_loader=data_loader,
                output_dir=str(tmp_path),
                config=config,
                sample_idx=0,
            )

        assert isinstance(artifact, Artifact)
        assert artifact.path == str(tmp_path)

        # Builder drove the decomposition off the extracted sample.
        extractor.extract_sample.assert_called_once_with(model, data_loader, 0)
        pipeline.component_builder.build_components.assert_called_once_with(model, "SAMPLE")

        assert exporter.export.call_count == 2
        written = {call.kwargs["output_path"]: call.kwargs for call in exporter.export.call_args_list}
        assert set(written) == {str(tmp_path / "voxel.onnx"), str(tmp_path / "head.onnx")}
        assert written[str(tmp_path / "voxel.onnx")]["model"] is voxel_module
        assert written[str(tmp_path / "voxel.onnx")]["sample_input"] == "VOXEL_IN"
        assert written[str(tmp_path / "head.onnx")]["model"] is head_module

    def test_component_export_failure_is_wrapped(self, tmp_path):
        exporter = Mock()
        exporter.export.side_effect = RuntimeError("boom")

        components = [ExportableComponent(name="model", module=object(), sample_input="IN")]
        extractor = Mock()
        extractor.extract_sample.return_value = "SAMPLE"

        pipeline = OnnxExportPipeline(
            sample_extractor=extractor,
            component_builder=_builder(components),
            onnx_wrapper_cls=Mock(),
        )
        config = _onnx_config({"model": SimpleNamespace(save_file="model.onnx", batch_size=None)})

        with patch(
            "deployment.export.pipelines.onnx_export_pipeline.ONNXExporter",
            return_value=exporter,
        ):
            with pytest.raises(RuntimeError, match="model ONNX export failed"):
                pipeline.export(
                    model=object(),
                    data_loader=Mock(),
                    output_dir=str(tmp_path),
                    config=config,
                    sample_idx=0,
                )

    def test_sample_extraction_failure_is_wrapped(self, tmp_path):
        extractor = Mock()
        extractor.extract_sample.side_effect = ValueError("no sample")

        pipeline = OnnxExportPipeline(
            sample_extractor=extractor,
            component_builder=Mock(),
        )

        with pytest.raises(RuntimeError, match="Sample extraction failed"):
            pipeline.export(
                model=object(),
                data_loader=Mock(),
                output_dir=str(tmp_path),
                config=_onnx_config({}),
                sample_idx=0,
            )


class TestDefaultSampleExtractor:
    def test_returns_preprocessed_loaded_sample(self):
        data_loader = Mock()
        data_loader.load_sample.return_value = "RAW"
        data_loader.preprocess.return_value = "PREPROCESSED"

        sample = DefaultSampleExtractor().extract_sample(object(), data_loader, 3)

        data_loader.load_sample.assert_called_once_with(3)
        data_loader.preprocess.assert_called_once_with("RAW")
        assert sample == "PREPROCESSED"


class TestDefaultComponentBuilder:
    def test_wraps_whole_model_as_single_component(self):
        components_cfg = SimpleNamespace(component_names=lambda: ["model"])
        model = object()

        components = DefaultComponentBuilder(components_cfg).build_components(model, "SAMPLE")

        assert len(components) == 1
        assert components[0].name == "model"
        assert components[0].module is model
        assert components[0].sample_input == "SAMPLE"

    def test_rejects_multi_component_config(self):
        """The whole-model builder cannot map onto a decomposed config."""
        components_cfg = SimpleNamespace(component_names=lambda: ["voxel", "head"])

        with pytest.raises(ValueError, match="exactly one component"):
            DefaultComponentBuilder(components_cfg).build_components(object(), "SAMPLE")


class TestTensorRTExportPipeline:
    """Guard the input validation that happens before any CUDA/TensorRT work."""

    def test_rejects_non_directory_onnx_path(self, tmp_path):
        pipeline = TensorRTExportPipeline()
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"x")

        with pytest.raises(ValueError, match="must be a directory"):
            pipeline.export(
                onnx_path=str(onnx_file),
                output_dir=str(tmp_path / "trt"),
                config=SimpleNamespace(components_cfg=SimpleNamespace(items=lambda: [])),
                device=SimpleNamespace(is_cuda=True, index=0),
            )

    def test_rejects_empty_components(self, tmp_path):
        pipeline = TensorRTExportPipeline()
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()

        with pytest.raises(ValueError, match="components config is empty"):
            pipeline.export(
                onnx_path=str(onnx_dir),
                output_dir=str(tmp_path / "trt"),
                config=SimpleNamespace(components_cfg=SimpleNamespace(items=lambda: [])),
                device=SimpleNamespace(is_cuda=True, index=0),
            )
