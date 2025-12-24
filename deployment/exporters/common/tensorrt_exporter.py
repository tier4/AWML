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
    ):
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
        self.logger.info(f"Building TensorRT engine with precision policy: {precision_policy}")
        self.logger.info(f"  ONNX source: {onnx_path}")
        self.logger.info(f"  Engine output: {output_path}")

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
                return Artifact(path=output_path, multi_file=False)
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
                self.logger.info(f"BuilderFlag.{flag_name} enabled")

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
        self._configure_input_shapes(profile, sample_input, network)
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
        self.logger.info(f"TensorRT engine saved to {output_path}")
        self.logger.info(f"Engine max workspace size: {max_workspace_size / (1024**3):.2f} GB")

    def _configure_input_shapes(
        self,
        profile: trt.IOptimizationProfile,
        sample_input: Any,
        network: trt.INetworkDefinition = None,
    ) -> None:
        """
        Configure input shapes for TensorRT optimization profile.

        Note:
            ONNX dynamic_axes is used for export; TRT profile is the runtime envelope;
            they are related but not equivalent.

            - **ONNX dynamic_axes**: Controls symbolic dimensions in the ONNX graph during
              export. Defines which dimensions can vary at runtime in the ONNX model.

            - **TensorRT profile (min/opt/max)**: Defines the runtime shape envelope that
              TensorRT optimizes for. TensorRT builds kernels optimized for shapes within
              this envelope. The profile must be compatible with the ONNX dynamic axes,
              but they are configured separately:
              - dynamic_axes: Export-time graph structure (what dimensions are variable)
              - TRT profile: Runtime optimization envelope (what shapes to optimize for)

            They are complementary but independent. The ONNX model may have dynamic axes,
            but TensorRT still needs explicit min/opt/max shapes to build optimized kernels.

        Raises:
            ValueError: If neither model_inputs config nor sample_input is provided
        """
        model_inputs_cfg = self.config.model_inputs

        # Validate that we have shape information
        first_input_shapes = None
        if model_inputs_cfg:
            first_input_shapes = self._extract_input_shapes(model_inputs_cfg[0])

        if not model_inputs_cfg or not first_input_shapes:
            if sample_input is None:
                raise ValueError(
                    "TensorRT export requires shape information. Please provide either:\n"
                    "  1. Explicit 'model_inputs' with 'input_shapes' (min/opt/max) in config, OR\n"
                    "  2. A 'sample_input' tensor for automatic shape inference\n"
                    "\n"
                    "Current config has:\n"
                    f"  - model_inputs: {model_inputs_cfg}\n"
                    f"  - sample_input: {sample_input}\n"
                    "\n"
                    "Example config:\n"
                    "  backend_config = dict(\n"
                    "      model_inputs=[\n"
                    "          dict(\n"
                    "              input_shapes={\n"
                    "                  'input': dict(\n"
                    "                      min_shape=(1, 3, 960, 960),\n"
                    "                      opt_shape=(1, 3, 960, 960),\n"
                    "                      max_shape=(1, 3, 960, 960),\n"
                    "                  )\n"
                    "              }\n"
                    "          )\n"
                    "      ]\n"
                    "  )"
                )
            # If we have sample_input but no config, we could infer shapes
            # For now, just require explicit config
            self.logger.warning(
                "sample_input provided but no explicit model_inputs config. "
                "TensorRT export may fail if ONNX has dynamic dimensions."
            )

        if not model_inputs_cfg:
            raise ValueError("model_inputs is not set in the config")

        # model_inputs is already a Tuple[TensorRTModelInputConfig, ...]
        first_entry = model_inputs_cfg[0]
        input_shapes = first_input_shapes

        if not input_shapes:
            raise ValueError("TensorRT model_inputs[0] missing 'input_shapes' definitions")

        for input_name, profile_cfg in input_shapes.items():
            min_shape, opt_shape, max_shape = self._resolve_profile_shapes(profile_cfg, sample_input, input_name)
            self.logger.info(f"Setting {input_name} shapes - min: {min_shape}, opt: {opt_shape}, max: {max_shape}")
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)

    def _log_parser_errors(self, parser: trt.OnnxParser) -> None:
        """Log TensorRT parser errors."""
        self.logger.error("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            self.logger.error(f"Parser error: {parser.get_error(error)}")

    def _extract_input_shapes(self, entry: Any) -> Mapping[str, Any]:
        if isinstance(entry, TensorRTModelInputConfig):
            return entry.input_shapes
        if isinstance(entry, Mapping):
            return entry.get("input_shapes", {}) or {}
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
