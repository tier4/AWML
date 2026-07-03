"""Unit tests for deploy-config schema parsing/validation.

These are pure-Python (no GPU) and exercise the validation branches in
``deployment.config.schema`` that previously had no coverage.
"""

from __future__ import annotations

import pytest

from deployment.config.schema import ComponentsConfig, VerificationScenario


def _valid_component() -> dict:
    return {
        "onnx_file": "model.onnx",
        "engine_file": "model.engine",
        "io": {
            "inputs": [{"name": "x", "dtype": "float32"}],
            "outputs": [{"name": "y", "dtype": "float32"}],
            "dynamic_axes": {"x": {0: "batch"}},
        },
    }


class TestComponentsConfig:
    def test_parses_valid_component(self):
        cfg = ComponentsConfig.from_dict({"model": _valid_component()})
        comp = cfg.get_component("model")
        assert comp.name == "model"
        assert comp.onnx_file == "model.onnx"
        assert comp.engine_file == "model.engine"
        assert [i.name for i in comp.io.inputs] == ["x"]
        assert [o.name for o in comp.io.outputs] == ["y"]
        assert comp.io.dynamic_axes == {"x": {0: "batch"}}

    def test_missing_onnx_file_raises(self):
        comp = _valid_component()
        del comp["onnx_file"]
        with pytest.raises(KeyError):
            ComponentsConfig.from_dict({"model": comp})

    def test_empty_outputs_raises(self):
        comp = _valid_component()
        comp["io"]["outputs"] = []
        with pytest.raises(KeyError):
            ComponentsConfig.from_dict({"model": comp})

    def test_bad_dynamic_axes_type_raises(self):
        comp = _valid_component()
        comp["io"]["dynamic_axes"] = {"x": {"not_an_int": "batch"}}
        with pytest.raises(TypeError):
            ComponentsConfig.from_dict({"model": comp})

    def test_unknown_component_lookup_raises(self):
        cfg = ComponentsConfig.from_dict({"model": _valid_component()})
        with pytest.raises(KeyError):
            cfg.get_component("does_not_exist")

    def test_get_artifact_filename(self):
        cfg = ComponentsConfig.from_dict({"model": _valid_component()})
        assert cfg.get_artifact_filename("model", "engine_file") == "model.engine"


class TestVerificationScenario:
    def test_parses_valid_scenario(self):
        scenario = VerificationScenario.from_dict(
            {
                "ref_backend": "pytorch",
                "ref_device": "cpu",
                "test_backend": "onnx",
                "test_device": "cpu",
            }
        )
        assert scenario.ref_backend.value == "pytorch"
        assert scenario.test_backend.value == "onnx"

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError):
            VerificationScenario.from_dict({"ref_backend": "pytorch"})
