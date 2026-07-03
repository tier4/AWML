"""
Unified ONNX export pipeline.

A single, model-agnostic pipeline drives every ONNX export. The only thing that
varies between models is *how the PyTorch model is split into exportable
components*, and that variation is injected via a :class:`ModelComponentBuilder`
(plus a :class:`SampleExtractor` that produces the tracing sample):

- Single-component models (e.g. a detector exported whole) use the built-in
  :class:`~deployment.export.pipelines.sample_extractor.DefaultSampleExtractor`
  and :class:`~deployment.export.pipelines.component_builder.DefaultComponentBuilder`.
- Models that must be decomposed (e.g. CenterPoint → voxel encoder +
  backbone/neck/head) provide a project-specific extractor and builder.

The pipeline itself never changes; it iterates whatever components the builder
returns and exports one ONNX file per component.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Type

import torch

from deployment.config.base import BaseDeploymentConfig
from deployment.export.exporters.model_wrappers import BaseModelWrapper, IdentityWrapper
from deployment.export.exporters.onnx_exporter import ONNXExporter
from deployment.export.pipelines.component_builder import ExportableComponent, ModelComponentBuilder
from deployment.export.pipelines.sample_extractor import SampleExtractor
from deployment.io.base_data_loader import BaseDataLoader
from deployment.primitives.artifacts import Artifact

logger = logging.getLogger(__name__)


class OnnxExportPipeline:
    """Model-agnostic ONNX export pipeline (one ONNX file per component).

    Extracts a tracing sample via ``sample_extractor``, splits the model into
    components via ``component_builder``, and exports each component with the
    configured ONNX exporter. The number of ONNX files produced is exactly the
    number of components the builder returns (one for a whole-model export,
    several for a decomposed model).
    """

    def __init__(
        self,
        sample_extractor: SampleExtractor,
        component_builder: ModelComponentBuilder,
        onnx_wrapper_cls: Type[BaseModelWrapper] = IdentityWrapper,
    ) -> None:
        """Initialize the pipeline.

        Args:
            sample_extractor: Extractor that produces the typed tracing sample.
            component_builder: Builder that turns the model + sample into
                exportable components.
            onnx_wrapper_cls: Model wrapper applied before ONNX export (defaults
                to ``IdentityWrapper`` for models needing no output reshaping).
        """
        self.sample_extractor = sample_extractor
        self.component_builder = component_builder
        self._onnx_wrapper_cls = onnx_wrapper_cls

    def export(
        self,
        *,
        model: torch.nn.Module,
        data_loader: BaseDataLoader,
        output_dir: str,
        config: BaseDeploymentConfig,
        sample_idx: int = 0,
    ) -> Artifact:
        """Export the model to ONNX, one file per builder-produced component.

        Args:
            model: PyTorch model to export.
            data_loader: Loader used to fetch the sample for tracing.
            output_dir: Directory where ONNX files are written.
            config: Deployment config for exporter options and component layout.
            sample_idx: Index of the sample to use for tracing (default 0).

        Returns:
            Artifact whose path is the output directory.

        Raises:
            RuntimeError: If sample extraction or a component export fails.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        sample = self._extract_sample(model, data_loader, sample_idx)
        components = self.component_builder.build_components(model, sample)

        logger.info("=" * 80)
        logger.info("Exporting %s component(s) to ONNX", len(components))
        logger.info("=" * 80)

        exported_paths = self._export_components(components, output_dir_path, config)
        self._log_summary(exported_paths)

        return Artifact(path=str(output_dir_path))

    def _extract_sample(self, model: torch.nn.Module, data_loader: BaseDataLoader, sample_idx: int) -> Any:
        """Extract the typed tracing sample, wrapping failures with context.

        Args:
            model: PyTorch model used for sample extraction.
            data_loader: Loader to fetch the sample from.
            sample_idx: Index of the sample.

        Returns:
            The extractor-specific sample payload consumed by the component builder.

        Raises:
            RuntimeError: If sample extraction fails.
        """
        logger.info("Extracting sample data (sample_idx=%s)...", sample_idx)
        try:
            return self.sample_extractor.extract_sample(model, data_loader, sample_idx)
        except Exception as exc:
            raise RuntimeError(f"Sample extraction failed: {exc}") from exc

    def _export_components(
        self,
        components: List[ExportableComponent],
        output_dir: Path,
        config: BaseDeploymentConfig,
    ) -> List[str]:
        """Export each component to its configured ONNX file under ``output_dir``.

        Args:
            components: Exportable components (name, module, sample_input).
            output_dir: Directory where ONNX files are written.
            config: Deployment config for exporter options and filenames.

        Returns:
            List of written ONNX file paths.

        Raises:
            RuntimeError: If any component export fails.
        """
        exported_paths: List[str] = []
        for index, component in enumerate(components, start=1):
            onnx_settings = config.get_onnx_settings(component.name)
            output_path = output_dir / onnx_settings.save_file
            output_path.parent.mkdir(parents=True, exist_ok=True)

            exporter = self._build_onnx_exporter(config, component.name)
            sample_input = self._apply_batch_size(component.sample_input, onnx_settings.batch_size)

            logger.info("\n[%s/%s] Exporting %s → %s", index, len(components), component.name, output_path)
            try:
                exporter.export(model=component.module, sample_input=sample_input, output_path=str(output_path))
            except Exception as exc:
                logger.error("Failed to export %s", component.name, exc_info=True)
                raise RuntimeError(f"{component.name} ONNX export failed") from exc

            exported_paths.append(str(output_path))

        return exported_paths

    def _build_onnx_exporter(self, config: BaseDeploymentConfig, component_name: str) -> ONNXExporter:
        """Create an ONNX exporter for the given component.

        Args:
            config: Deployment config used to construct the ONNX exporter.
            component_name: Component name used to resolve component-level options.

        Returns:
            Configured ONNX exporter for the target component.
        """
        return ONNXExporter(
            config=config.get_onnx_settings(component_name),
            model_wrapper=self._onnx_wrapper_cls,
        )

    def _log_summary(self, exported_paths: List[str]) -> None:
        """Log a success summary listing the exported ONNX files.

        Args:
            exported_paths: Paths of successfully exported ONNX files.
        """
        logger.info("\n" + "=" * 80)
        logger.info("ONNX export successful (%s file(s))", len(exported_paths))
        logger.info("=" * 80)
        for path in exported_paths:
            logger.info("  • %s", Path(path).name)

    @staticmethod
    def _apply_batch_size(sample_input: Any, batch_size: Optional[int]) -> Any:
        """Repeat a single-sample input along the batch dimension if requested.

        Args:
            sample_input: Preprocessed input (tensor, or list/tuple of tensors).
            batch_size: Target batch size, or None to leave the input unchanged.

        Returns:
            The input repeated to ``batch_size`` along dim 0, or unchanged if
            ``batch_size`` is None.
        """
        if batch_size is None:
            return sample_input
        if isinstance(sample_input, (list, tuple)):
            return tuple(
                inp.repeat(batch_size, *([1] * (len(inp.shape) - 1))) if len(inp.shape) > 0 else inp
                for inp in sample_input
            )
        return sample_input.repeat(batch_size, *([1] * (len(sample_input.shape) - 1)))
