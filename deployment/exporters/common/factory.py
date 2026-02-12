"""
Factory helpers for creating exporter instances from deployment configs.
"""

from __future__ import annotations

import logging
from typing import Optional, Type

from deployment.core import BaseDeploymentConfig
from deployment.exporters.common.configs import TensorRTExportConfig
from deployment.exporters.common.model_wrappers import BaseModelWrapper
from deployment.exporters.common.onnx_exporter import ONNXExporter
from deployment.exporters.common.tensorrt_exporter import TensorRTExporter


class ExporterFactory:
    """
    Factory class for instantiating exporters using deployment configs.
    """

    @staticmethod
    def create_onnx_exporter(
        config: BaseDeploymentConfig,
        wrapper_cls: Type[BaseModelWrapper],
        logger: logging.Logger,
    ) -> ONNXExporter:
        """
        Build an ONNX exporter using the deployment config settings.
        """

        return ONNXExporter(
            config=config.get_onnx_settings(),
            model_wrapper=wrapper_cls,
            logger=logger,
        )

    @staticmethod
    def create_tensorrt_exporter(
        config: BaseDeploymentConfig,
        logger: logging.Logger,
        config_override: Optional[TensorRTExportConfig] = None,
    ) -> TensorRTExporter:
        """
        Build a TensorRT exporter using the deployment config settings.

        Args:
            config: Deployment configuration
            logger: Logger instance
            config_override: Optional TensorRT config to use instead of the one
                           derived from the deployment config. Useful for
                           per-component configurations in multi-file exports.
        """
        trt_config = config_override if config_override is not None else config.get_tensorrt_settings()

        return TensorRTExporter(
            config=trt_config,
            logger=logger,
        )
