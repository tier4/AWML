"""TensorRT model exporter."""

import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import tensorrt as trt
import torch

from deployment.core.artifacts import Artifact
from deployment.exporters.common.base_exporter import BaseExporter
from deployment.exporters.common.configs import TensorRTExportConfig, TensorRTModelInputConfig, TensorRTProfileConfig


class TensorRTExporter(BaseExporter):
    """
    TensorRT model exporter.

    Converts ONNX models to TensorRT engine format with precision policy support.
    """

    def __init__(
        self,
        config: TensorRTExportConfig,
        model_wrapper: Optional[Any] = None,
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize TensorRT exporter.

        Args:
            config: TensorRT export configuration dataclass instance.
            model_wrapper: Optional model wrapper class (usually not needed for TensorRT)
            logger: Optional logger instance
        """
        super().__init__(config, model_wrapper=model_wrapper, logger=logger)
        self.logger = logger or logging.getLogger(__name__)

    def export(
        self,
        model: torch.nn.Module,  # Not used for TensorRT, kept for interface compatibility
        sample_input: Any,
        output_path: str,
        onnx_path: str = None,
    ) -> Artifact:
        """
        Export ONNX model to TensorRT engine.

        Args:
            model: Not used (TensorRT converts from ONNX)
            sample_input: Sample input for shape configuration
            output_path: Path to save TensorRT engine
            onnx_path: Path to source ONNX model

        Returns:
            Artifact object representing the exported TensorRT engine

        Raises:
            RuntimeError: If export fails
            ValueError: If ONNX path is missing
        """
        if onnx_path is None:
            raise ValueError("onnx_path is required for TensorRT export")

        precision_policy = self.config.precision_policy
        self.logger.info("Building TensorRT engine with precision policy: %s", precision_policy)
        self.logger.info("  ONNX source: %s", onnx_path)
        self.logger.info("  Engine output: %s", output_path)

        return self._do_tensorrt_export(onnx_path, output_path, sample_input)

    def _do_tensorrt_export(
        self,
        onnx_path: str,
        output_path: str,
        sample_input: Any,
    ) -> Artifact:
        """
        Export a single ONNX file to TensorRT engine.

        This method handles the complete export workflow with proper resource management.

        Args:
            onnx_path: Path to source ONNX model
            output_path: Path to save TensorRT engine
            sample_input: Sample input for shape configuration

        Returns:
            Artifact object representing the exported TensorRT engine

        Raises:
            RuntimeError: If export fails
        """
        # Initialize TensorRT
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, "")

        builder = trt.Builder(trt_logger)
        try:
            builder_config, network, parser = self._create_builder_and_network(builder, trt_logger)
            try:
                self._parse_onnx(parser, network, onnx_path)
                self._configure_input_profiles(builder, builder_config, network, sample_input)
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

        # Create network with appropriate flags
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        # Handle strongly typed flag (network creation flag)
        policy_flags = self.config.policy_flags
        if policy_flags.get("STRONGLY_TYPED", False):
            flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            self.logger.info("Using strongly typed TensorRT network creation")

        network = builder.create_network(flags)

        # Apply precision flags to builder config
        for flag_name, enabled in policy_flags.items():
            if flag_name == "STRONGLY_TYPED":
                continue
            if enabled and hasattr(trt.BuilderFlag, flag_name):
                builder_config.set_flag(getattr(trt.BuilderFlag, flag_name))
                self.logger.info("BuilderFlag.%s enabled", flag_name)

        parser = trt.OnnxParser(network, trt_logger)

        return builder_config, network, parser

    def _parse_onnx(
        self,
        parser: trt.OnnxParser,
        network: trt.INetworkDefinition,
        onnx_path: str,
    ) -> None:
        """
        Parse ONNX model into TensorRT network.

        Args:
            parser: TensorRT ONNX parser instance
            network: TensorRT network definition
            onnx_path: Path to ONNX model file

        Raises:
            RuntimeError: If parsing fails
        """
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                self._log_parser_errors(parser)
                raise RuntimeError("TensorRT export failed: unable to parse ONNX file")
        self.logger.info("Successfully parsed ONNX file")

    def _configure_input_profiles(
        self,
        builder: trt.Builder,
        builder_config: trt.IBuilderConfig,
        network: trt.INetworkDefinition,
        sample_input: Any,
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
            network: TensorRT network definition
            sample_input: Sample input for shape configuration (typically obtained via
                         BaseDataLoader.get_shape_sample())
        """
        profile = builder.create_optimization_profile()
        self._configure_input_shapes(profile, sample_input)
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
        self.logger.info("Building TensorRT engine (this may take a while)...")
        serialized_engine = builder.build_serialized_network(network, builder_config)

        if serialized_engine is None:
            self.logger.error("Failed to build TensorRT engine")
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
        with open(output_path, "wb") as f:
            f.write(serialized_engine)

        max_workspace_size = self.config.max_workspace_size
        self.logger.info("TensorRT engine saved to %s", output_path)
        self.logger.info("Engine max workspace size: %.2f GB", max_workspace_size / (1024**3))

    def _configure_input_shapes(
        self,
        profile: trt.IOptimizationProfile,
        sample_input: Any,
    ) -> None:
        """Configure TensorRT optimization profile shapes from config."""
        model_inputs_cfg = self.config.model_inputs
        if not model_inputs_cfg:
            if sample_input is not None:
                self.logger.warning(
                    "sample_input was provided, but no explicit model_inputs config was found. "
                    "TensorRT export may fail if the ONNX model contains dynamic dimensions."
                )
            raise ValueError(self._build_missing_model_inputs_error(sample_input))

        input_shapes = self._extract_input_shapes(model_inputs_cfg[0])
        if not input_shapes:
            raise ValueError("TensorRT model_inputs[0] must define 'input_shapes'.")

        for input_name, profile_cfg in input_shapes.items():
            min_shape, opt_shape, max_shape = self._resolve_profile_shapes(
                profile_cfg=profile_cfg,
                sample_input=sample_input,
                input_name=input_name,
            )
            self.logger.info(
                "Setting %s shapes - min: %s, opt: %s, max: %s",
                input_name,
                min_shape,
                opt_shape,
                max_shape,
            )
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)

    def _log_parser_errors(self, parser: trt.OnnxParser) -> None:
        """Log TensorRT parser errors."""
        self.logger.error("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            self.logger.error("Parser error: %s", parser.get_error(error))

    def _extract_input_shapes(self, entry: Any) -> Mapping[str, Any]:
        if isinstance(entry, TensorRTModelInputConfig):
            return entry.input_shapes
        if isinstance(entry, Mapping):
            input_shapes = entry.get("input_shapes")
            if input_shapes is None:
                input_shapes = {}
            if not isinstance(input_shapes, Mapping):
                raise TypeError(f"input_shapes must be a mapping, got {type(input_shapes).__name__}")
            return input_shapes
        raise TypeError(f"Unsupported TensorRT model input entry: {type(entry)}")

    def _resolve_profile_shapes(
        self,
        profile_cfg: Any,
        sample_input: Any,
        input_name: str,
    ) -> Sequence[Sequence[int]]:
        if isinstance(profile_cfg, TensorRTProfileConfig):
            min_shape = self._shape_to_list(profile_cfg.min_shape)
            opt_shape = self._shape_to_list(profile_cfg.opt_shape)
            max_shape = self._shape_to_list(profile_cfg.max_shape)
        elif isinstance(profile_cfg, Mapping):
            min_shape = self._shape_to_list(profile_cfg.get("min_shape"))
            opt_shape = self._shape_to_list(profile_cfg.get("opt_shape"))
            max_shape = self._shape_to_list(profile_cfg.get("max_shape"))
        else:
            raise TypeError(f"Unsupported TensorRT profile type for input '{input_name}': {type(profile_cfg)}")

        return (
            self._ensure_shape(min_shape, sample_input, input_name, "min"),
            self._ensure_shape(opt_shape, sample_input, input_name, "opt"),
            self._ensure_shape(max_shape, sample_input, input_name, "max"),
        )

    @staticmethod
    def _shape_to_list(shape: Optional[Sequence[int]]) -> Optional[Sequence[int]]:
        if shape is None:
            return None
        return [int(dim) for dim in shape]

    def _ensure_shape(
        self,
        shape: Optional[Sequence[int]],
        sample_input: Any,
        input_name: str,
        bucket: str,
    ) -> Sequence[int]:
        if shape:
            return list(shape)
        if sample_input is None or not hasattr(sample_input, "shape"):
            raise ValueError(f"{bucket}_shape missing for {input_name} and sample_input is not provided")
        inferred = list(sample_input.shape)
        self.logger.debug("Falling back to sample_input.shape=%s for %s:%s", inferred, input_name, bucket)
        return inferred
