from __future__ import annotations

import logging
from typing import Type

from deployment.core import BaseDeploymentConfig
from deployment.exporters.common.model_wrappers import BaseModelWrapper
from deployment.exporters.common.onnx_exporter import ONNXExporter
from deployment.exporters.common.tensorrt_exporter import TensorRTExporter


class ExporterFactory:
    """Factory class for instantiating exporters using deployment configs."""

    @staticmethod
    def create_onnx_exporter(
        config: BaseDeploymentConfig,
        wrapper_cls: Type[BaseModelWrapper],
        logger: logging.Logger,
        component_name: str,
    ) -> ONNXExporter:
        """Build an ONNX exporter for the given component."""
        return ONNXExporter(
            config=config.get_onnx_settings(component_name),
            model_wrapper=wrapper_cls,
            logger=logger,
        )

    @staticmethod
    def create_tensorrt_exporter(
        config: BaseDeploymentConfig,
        logger: logging.Logger,
        component_name: str,
    ) -> TensorRTExporter:
        """Build a TensorRT exporter for the given component."""
        return TensorRTExporter(
            config=config.get_tensorrt_settings(component_name),
            logger=logger,
        )
