"""
CenterPoint ONNX export pipeline using composition.

Splits the CenterPoint model into exportable components (e.g. voxel encoder,
backbone+neck+head) via composition and exports each component
to a separate ONNX file in the given output directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import torch
from typing_extensions import override

from deployment.configs.base import BaseDeploymentConfig
from deployment.core.artifacts import Artifact
from deployment.core.io.base_data_loader import BaseDataLoader
from deployment.exporters.common.factory import ExporterFactory
from deployment.exporters.common.model_wrappers import IdentityWrapper
from deployment.exporters.common.onnx_exporter import ONNXExporter
from deployment.exporters.export_pipelines.base import OnnxExportPipeline
from deployment.exporters.export_pipelines.interfaces import (
    ExportableComponent,
    ExportSampleAdapter,
    ModelComponentBuilder,
)
from deployment.projects.centerpoint.io.sample_types import CenterPointFeatureSample


class CenterPointONNXExportPipeline(OnnxExportPipeline):
    """ONNX export pipeline for CenterPoint (multi-file export).

    Uses a sample adapter + component builder to split the model into exportable
    components and exports each with the configured ONNX exporter.
    """

    def __init__(
        self,
        exporter_factory: type[ExporterFactory],
        sample_adapter: ExportSampleAdapter,
        component_builder: ModelComponentBuilder,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the pipeline with exporter factory, adapter, and builder.

        Args:
            exporter_factory: Factory used to create ONNX exporters per component.
            sample_adapter: Adapter that extracts typed sample payload.
            component_builder: Builder that creates exportable components from sample.
            logger: Optional logger; defaults to module logger if not provided.
        """
        self.exporter_factory = exporter_factory
        self.sample_adapter = sample_adapter
        self.component_builder = component_builder
        self.logger = logger or logging.getLogger(__name__)

    @override
    def export(
        self,
        *,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        output_dir: str,
        config: BaseDeploymentConfig,
        sample_idx: int = 0,
    ) -> Artifact:
        """Export CenterPoint to multi-file ONNX (one file per component).

        Extracts sample data, splits the model into components, and exports each
        component to ``<output_dir>/<component_name>.onnx``.

        Args:
            model: CenterPoint model to export.
            data_loader: Loader used to get sample data for tracing.
            output_dir: Directory where ONNX files are written.
            config: Deployment config for exporter options.
            sample_idx: Index of the sample to use for export (default 0).

        Returns:
            Artifact whose path is the output directory.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        self._log_header(output_dir_path, sample_idx)
        sample = self._extract_sample_data(model, data_loader, sample_idx)
        components = self.component_builder.build_components(model, sample)

        exported_paths = self._export_components(components, output_dir_path, config)
        self._log_summary(exported_paths)

        return Artifact(path=str(output_dir_path))

    def _log_header(self, output_dir: Path, sample_idx: int) -> None:
        """Log export header with output directory and sample index.

        Args:
            output_dir: Directory where exported ONNX files are written.
            sample_idx: Index of sample used for tracing/export.
        """
        self.logger.info("=" * 80)
        self.logger.info("Exporting CenterPoint to ONNX (multi-file)")
        self.logger.info("=" * 80)
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Using sample index: {sample_idx}")

    def _extract_sample_data(
        self,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        sample_idx: int,
    ) -> CenterPointFeatureSample:
        """Extract typed sample payload for component building.

        Args:
            model: CenterPoint model (must have _extract_features for ONNX export).
            data_loader: Loader to fetch the sample from.
            sample_idx: Index of the sample.

        Returns:
            Typed `CenterPointFeatureSample` payload.

        Raises:
            RuntimeError: If feature extraction fails.
        """
        self.logger.info("Extracting features from sample data...")
        try:
            return self.sample_adapter.extract_sample(model, data_loader, sample_idx)
        except Exception as exc:
            self.logger.error("Failed to extract features", exc_info=exc)
            raise RuntimeError("Feature extraction failed") from exc

    def _export_components(
        self,
        components: Iterable[ExportableComponent],
        output_dir: Path,
        config: BaseDeploymentConfig,
    ) -> list[str]:
        """Export each component to ONNX under output_dir (one file per component).

        Args:
            components: Exportable components (name, module, sample_input).
            output_dir: Directory to write <component.name>.onnx files.
            config: Deployment config for building the ONNX exporter.

        Returns:
            List of absolute paths of exported ONNX files.

        Raises:
            RuntimeError: If any component export fails.
        """
        exported_paths: list[str] = []
        component_list = list(components)
        total = len(component_list)

        for index, component in enumerate(component_list, start=1):
            self.logger.info(f"\n[{index}/{total}] Exporting {component.name}...")
            output_path = output_dir / f"{component.name}.onnx"
            exporter = self._build_onnx_exporter(config, component_name=component.name)

            try:
                exporter.export(
                    model=component.module,
                    sample_input=component.sample_input,
                    output_path=str(output_path),
                )
            except Exception as exc:
                self.logger.error(f"Failed to export {component.name}", exc_info=exc)
                raise RuntimeError(f"{component.name} export failed") from exc

            exported_paths.append(str(output_path))
            self.logger.info(f"Exported {component.name}: {output_path}")

        return exported_paths

    def _build_onnx_exporter(self, config: BaseDeploymentConfig, component_name: str) -> ONNXExporter:
        """Create an ONNX exporter for the given component using the factory.

        Args:
            config: Deployment config used to construct the ONNX exporter.
            component_name: Component name used to resolve component-level options.

        Returns:
            Configured ONNX exporter for the target component.
        """
        return self.exporter_factory.create_onnx_exporter(
            config=config,
            wrapper_cls=IdentityWrapper,
            logger=self.logger,
            component_name=component_name,
        )

    def _log_summary(self, exported_paths: list[str]) -> None:
        """Log success summary and list of exported ONNX file paths.

        Args:
            exported_paths: Paths of successfully exported ONNX files.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CenterPoint ONNX export successful")
        self.logger.info("=" * 80)
        for path in exported_paths:
            self.logger.info(f"  • {os.path.basename(path)}")
