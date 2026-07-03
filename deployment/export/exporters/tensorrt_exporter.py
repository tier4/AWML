"""TensorRT model exporter."""

import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import tensorrt as trt

from deployment.config.enums import PrecisionPolicy
from deployment.export.exporters.configs import TensorRTExportConfig
from deployment.primitives.artifacts import Artifact

logger = logging.getLogger(__name__)


class TensorRTExporter:
    """
    TensorRT model exporter.

    Converts ONNX models to TensorRT engine format with precision policy support.
    """

    def __init__(
        self,
        config: TensorRTExportConfig,
    ) -> None:
        """
        Initialize TensorRT exporter.

        Args:
            config: TensorRT export configuration dataclass instance.
        """
        self.config = config

    def export(
        self,
        onnx_path: str,
        output_path: str,
    ) -> Artifact:
        """
        Export ONNX model to TensorRT engine.

        Args:
            onnx_path: Path to source ONNX model.
            output_path: Path to save TensorRT engine.

        Returns:
            Artifact object representing the exported TensorRT engine

        Raises:
            RuntimeError: If export fails
        """
        logger.info("Building TensorRT engine with precision policy: %s", self.config.precision_policy.value)
        logger.info("  ONNX source: %s", onnx_path)
        logger.info("  Engine output: %s", output_path)

        # Initialize TensorRT
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, "")

        builder = trt.Builder(trt_logger)
        try:
            builder_config, network, parser = self._create_builder_and_network(builder, trt_logger)
            try:
                self._parse_onnx(parser, onnx_path)
                self._configure_input_profiles(builder, builder_config)
                serialized_engine = self._build_engine(builder, builder_config, network)
                self._save_engine(serialized_engine, output_path)
                return Artifact(path=output_path)
            finally:
                del parser
                del network
        finally:
            del builder

    def _create_builder_and_network(
        self,
        builder: trt.Builder,
        trt_logger: trt.Logger,
    ) -> Tuple[trt.IBuilderConfig, trt.INetworkDefinition, trt.OnnxParser]:
        """
        Create builder config, network, and parser.

        Args:
            builder: TensorRT builder instance
            trt_logger: TensorRT logger instance

        Returns:
            Tuple of (builder_config, network, parser)
        """
        builder_config = builder.create_builder_config()

        max_workspace_size = self.config.max_workspace_size
        builder_config.set_memory_pool_limit(pool=trt.MemoryPoolType.WORKSPACE, pool_size=max_workspace_size)

        # EXPLICIT_BATCH plus any network-creation flags the precision policy needs.
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network_flags = self._apply_precision_policy(network_flags, builder_config)

        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, trt_logger)

        return builder_config, network, parser

    def _apply_precision_policy(self, network_flags: int, builder_config: trt.IBuilderConfig) -> int:
        """Apply the configured precision policy to TensorRT.

        Returns the (possibly updated) network-creation flags. ``STRONGLY_TYPED`` is a
        network-creation flag and must be folded in before the network is created;
        ``FP16``/``TF32`` are builder flags set on the builder config. ``AUTO`` adds nothing
        and lets TensorRT decide.
        """
        policy = self.config.precision_policy
        if policy is PrecisionPolicy.STRONGLY_TYPED:
            network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            logger.info("Using strongly typed TensorRT network creation")
        elif policy is PrecisionPolicy.FP16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            logger.info("BuilderFlag.FP16 enabled")
        elif policy is PrecisionPolicy.FP32_TF32:
            builder_config.set_flag(trt.BuilderFlag.TF32)
            logger.info("BuilderFlag.TF32 enabled")
        return network_flags

    def _parse_onnx(
        self,
        parser: trt.OnnxParser,
        onnx_path: str,
    ) -> None:
        """
        Parse ONNX model into the TensorRT network bound to `parser`.

        Args:
            parser: TensorRT ONNX parser instance
            onnx_path: Path to ONNX model file

        Raises:
            RuntimeError: If parsing fails
        """
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                self._log_parser_errors(parser)
                raise RuntimeError("TensorRT export failed: unable to parse ONNX file")
        logger.info("Successfully parsed ONNX file")

    def _configure_input_profiles(
        self,
        builder: trt.Builder,
        builder_config: trt.IBuilderConfig,
    ) -> None:
        """
        Configure TensorRT optimization profiles for input shapes.

        Creates an optimization profile and configures min/opt/max shapes for each input.
        See `_configure_input_shapes` for details on shape configuration.

        Note:
            ONNX `dynamic_axes` and TensorRT profiles serve different purposes:

            - **ONNX dynamic_axes**: Used during ONNX export to define which dimensions
              are symbolic (dynamic) in the ONNX graph. This allows the ONNX model to
              accept inputs of varying sizes at those dimensions.

            - **TensorRT profile**: Defines the runtime shape envelope (min/opt/max) that
              TensorRT will optimize for. TensorRT builds kernels optimized for shapes
              within this envelope. The profile must be compatible with the ONNX dynamic
              axes, but they are configured separately and serve different roles:
              - dynamic_axes: Export-time graph structure
              - TRT profile: Runtime optimization envelope

            They are related but not equivalent. The ONNX model may have dynamic axes,
            but TensorRT still needs explicit min/opt/max shapes to build optimized kernels.

        Args:
            builder: TensorRT builder instance
            builder_config: TensorRT builder config
        """
        profile = builder.create_optimization_profile()
        self._configure_input_shapes(profile)
        builder_config.add_optimization_profile(profile)

    def _build_engine(
        self,
        builder: trt.Builder,
        builder_config: trt.IBuilderConfig,
        network: trt.INetworkDefinition,
    ) -> bytes:
        """
        Build TensorRT engine from network.

        Args:
            builder: TensorRT builder instance
            builder_config: TensorRT builder config
            network: TensorRT network definition

        Returns:
            Serialized engine as bytes

        Raises:
            RuntimeError: If engine building fails
        """
        logger.info("Building TensorRT engine (this may take a while)...")
        serialized_engine = builder.build_serialized_network(network, builder_config)

        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            raise RuntimeError("TensorRT export failed: builder returned None")

        return serialized_engine

    def _save_engine(
        self,
        serialized_engine: bytes,
        output_path: str,
    ) -> None:
        """
        Save serialized TensorRT engine to file.

        Args:
            serialized_engine: Serialized engine bytes
            output_path: Path to save engine file
        """
        # Write to a temp file in the same directory, then atomically replace the target.
        # Building an engine can take many minutes; a crash mid-write must not leave a
        # truncated .engine that a later run would treat as a valid artifact.
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output.with_name(f"{output.name}.tmp")
        try:
            with open(tmp_path, "wb") as f:
                f.write(serialized_engine)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, output)
        finally:
            tmp_path.unlink(missing_ok=True)

        max_workspace_size = self.config.max_workspace_size
        logger.info("TensorRT engine saved to %s", output_path)
        logger.info("Engine max workspace size: %.2f GB", max_workspace_size / (1024**3))

    def _configure_input_shapes(
        self,
        profile: trt.IOptimizationProfile,
    ) -> None:
        """Configure TensorRT optimization profile shapes from config."""
        model_input_cfg = self.config.model_input
        if model_input_cfg is None or not model_input_cfg.input_shapes:
            raise ValueError(
                "TensorRT export requires 'model_input' with 'input_shapes' (min/opt/max per "
                "input tensor), but none were configured."
            )

        for input_name, profile_cfg in model_input_cfg.input_shapes.items():
            min_shape = self._to_int_list(profile_cfg.min_shape, input_name, "min")
            opt_shape = self._to_int_list(profile_cfg.opt_shape, input_name, "opt")
            max_shape = self._to_int_list(profile_cfg.max_shape, input_name, "max")
            logger.info(
                "Setting %s shapes - min: %s, opt: %s, max: %s",
                input_name,
                min_shape,
                opt_shape,
                max_shape,
            )
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)

    def _log_parser_errors(self, parser: trt.OnnxParser) -> None:
        """Log TensorRT parser errors."""
        logger.error("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            logger.error("Parser error: %s", parser.get_error(error))

    @staticmethod
    def _to_int_list(shape: Sequence[int], input_name: str, bucket: str) -> List[int]:
        """Coerce a configured profile shape to a list of ints; fail loud if missing."""
        if not shape:
            raise ValueError(f"{bucket}_shape missing for TensorRT input '{input_name}'.")
        return [int(dim) for dim in shape]
