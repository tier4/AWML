"""
Unified deployment runner for common deployment workflows.
This module provides a unified runner that handles the common deployment workflow
across different projects, while allowing project-specific customization.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from mmengine.config import Config

from autoware_ml.deployment.core import BaseDataLoader, BaseDeploymentConfig, BaseEvaluator
from autoware_ml.deployment.exporters.onnx_exporter import ONNXExporter
from autoware_ml.deployment.exporters.tensorrt_exporter import TensorRTExporter


class DeploymentRunner:
    """
    Unified deployment runner for common deployment workflows.

    This runner handles the standard deployment workflow:
    1. Load PyTorch model (if needed)
    2. Export to ONNX (if requested)
    3. Export to TensorRT (if requested)
    4. Verify outputs (if enabled)
    5. Evaluate models (if enabled)

    Projects can customize behavior by:
    - Overriding methods (load_pytorch_model, export_onnx, export_tensorrt)
    - Providing custom callbacks
    - Extending this class
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        evaluator: BaseEvaluator,
        config: BaseDeploymentConfig,
        model_cfg: Config,
        logger: logging.Logger,
        load_model_fn: Optional[Callable] = None,
        export_onnx_fn: Optional[Callable] = None,
        export_tensorrt_fn: Optional[Callable] = None,
        onnx_exporter: Optional[Any] = None,
        tensorrt_exporter: Optional[Any] = None,
    ):
        """
        Initialize unified deployment runner.

        Args:
            data_loader: Data loader for samples
            evaluator: Evaluator for model evaluation
            config: Deployment configuration
            model_cfg: Model configuration
            logger: Logger instance
            load_model_fn: Optional custom function to load PyTorch model
            export_onnx_fn: Optional custom function to export ONNX
            export_tensorrt_fn: Optional custom function to export TensorRT
            onnx_exporter: Optional ONNX exporter instance (e.g., CenterPointONNXExporter)
            tensorrt_exporter: Optional TensorRT exporter instance (e.g., CenterPointTensorRTExporter)
        """
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.config = config
        self.model_cfg = model_cfg
        self.logger = logger
        self._load_model_fn = load_model_fn
        self._export_onnx_fn = export_onnx_fn
        self._export_tensorrt_fn = export_tensorrt_fn
        self._onnx_exporter = onnx_exporter
        self._tensorrt_exporter = tensorrt_exporter

    def load_pytorch_model(self, checkpoint_path: str, **kwargs) -> Any:
        """
        Load PyTorch model from checkpoint.

        Uses custom function if provided, otherwise uses default implementation.

        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional project-specific arguments

        Returns:
            Loaded PyTorch model
        """
        if self._load_model_fn:
            return self._load_model_fn(checkpoint_path, **kwargs)

        # Default implementation - should be overridden by projects
        self.logger.warning("Using default load_pytorch_model - projects should override this")
        raise NotImplementedError("load_pytorch_model must be implemented or provided via load_model_fn")

    def export_onnx(self, pytorch_model: Any, **kwargs) -> Optional[str]:
        """
        Export model to ONNX format.

        Uses custom function if provided, otherwise uses standard ONNXExporter.

        Args:
            pytorch_model: PyTorch model to export
            **kwargs: Additional project-specific arguments

        Returns:
            Path to exported ONNX file/directory, or None if export failed
        """
        if self._export_onnx_fn:
            return self._export_onnx_fn(pytorch_model, self.data_loader, self.config, self.logger, **kwargs)

        # Standard ONNX export using ONNXExporter
        if not self.config.export_config.should_export_onnx():
            return None

        self.logger.info("=" * 80)
        self.logger.info("Exporting to ONNX (Using Unified ONNXExporter)")
        self.logger.info("=" * 80)

        # Get ONNX settings
        onnx_settings = self.config.get_onnx_settings()

        # Use provided exporter instance if available
        if self._onnx_exporter is not None:
            exporter = self._onnx_exporter
            self.logger.info("=" * 80)
            self.logger.info(f"Exporting to ONNX (Using {type(exporter).__name__})")
            self.logger.info("=" * 80)

            # Check if it's a CenterPoint exporter (needs special handling)
            # CenterPoint exporter has data_loader parameter in export method
            import inspect

            sig = inspect.signature(exporter.export)
            if "data_loader" in sig.parameters:
                # CenterPoint exporter signature
                # Save to work_dir/onnx/ directory
                output_dir = os.path.join(self.config.export_config.work_dir, "onnx")
                os.makedirs(output_dir, exist_ok=True)
                if not hasattr(pytorch_model, "_extract_features"):
                    self.logger.error("❌ ONNX export requires an ONNX-compatible model (CenterPointONNX).")
                    return None

                success = exporter.export(
                    model=pytorch_model, data_loader=self.data_loader, output_dir=output_dir, sample_idx=0
                )

                if success:
                    self.logger.info(f"✅ ONNX export successful: {output_dir}")
                    return output_dir
                else:
                    self.logger.error(f"❌ ONNX export failed")
                    return None

        # Standard ONNX export
        # Save to work_dir/onnx/ directory
        onnx_dir = os.path.join(self.config.export_config.work_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        output_path = os.path.join(onnx_dir, onnx_settings["save_file"])

        # Get sample input
        sample_idx = self.config.runtime_config.get("sample_idx", 0)
        sample = self.data_loader.load_sample(sample_idx)
        single_input = self.data_loader.preprocess(sample)

        # Ensure tensor is float32
        if single_input.dtype != torch.float32:
            single_input = single_input.float()

        # Get batch size from configuration
        batch_size = onnx_settings.get("batch_size", 1)
        if batch_size is None:
            input_tensor = single_input
            self.logger.info("Using dynamic batch size")
        else:
            # Handle different input shapes
            if isinstance(single_input, (list, tuple)):
                # Multiple inputs
                input_tensor = tuple(
                    inp.repeat(batch_size, *([1] * (len(inp.shape) - 1))) if len(inp.shape) > 0 else inp
                    for inp in single_input
                )
            else:
                # Single input
                input_tensor = single_input.repeat(batch_size, *([1] * (len(single_input.shape) - 1)))
            self.logger.info(f"Using fixed batch size: {batch_size}")

        # Use provided exporter or create default
        if self._onnx_exporter is not None:
            exporter = self._onnx_exporter
        else:
            exporter = ONNXExporter(onnx_settings, self.logger)

        success = exporter.export(pytorch_model, input_tensor, output_path)

        if success:
            self.logger.info(f"✅ ONNX export successful: {output_path}")
            return output_path
        else:
            self.logger.error(f"❌ ONNX export failed")
            return None

    def export_tensorrt(self, onnx_path: str, **kwargs) -> Optional[str]:
        """
        Export ONNX model to TensorRT engine.

        Uses custom function if provided, otherwise uses standard TensorRTExporter.

        Args:
            onnx_path: Path to ONNX model file/directory
            **kwargs: Additional project-specific arguments

        Returns:
            Path to exported TensorRT engine file/directory, or None if export failed
        """
        if self._export_tensorrt_fn:
            return self._export_tensorrt_fn(onnx_path, self.config, self.data_loader, self.logger, **kwargs)

        # Standard TensorRT export using TensorRTExporter
        if not self.config.export_config.should_export_tensorrt():
            return None

        if not onnx_path:
            self.logger.warning("ONNX path not available, skipping TensorRT export")
            return None

        trt_settings = self.config.get_tensorrt_settings()

        # Use provided exporter instance if available
        if self._tensorrt_exporter is not None:
            exporter = self._tensorrt_exporter
            self.logger.info("=" * 80)
            self.logger.info(f"Exporting to TensorRT (Using {type(exporter).__name__})")
            self.logger.info("=" * 80)

            # Check if it's a CenterPoint exporter (needs special handling)
            # CenterPoint exporter has onnx_dir parameter in export method
            import inspect

            sig = inspect.signature(exporter.export)
            if "onnx_dir" in sig.parameters:
                # CenterPoint exporter signature
                if not os.path.isdir(onnx_path):
                    self.logger.error("CenterPoint requires ONNX directory, not a single file")
                    return None

                # Save to work_dir/tensorrt/ directory
                output_dir = os.path.join(self.config.export_config.work_dir, "tensorrt")
                os.makedirs(output_dir, exist_ok=True)

                success = exporter.export(
                    onnx_dir=onnx_path, output_dir=output_dir, device=self.config.export_config.device
                )

                if success:
                    self.logger.info(f"✅ TensorRT export successful: {output_dir}")
                    return output_dir
                else:
                    self.logger.error(f"❌ TensorRT export failed")
                    return None

        # Standard TensorRT export
        self.logger.info("=" * 80)
        self.logger.info("Exporting to TensorRT (Using Unified TensorRTExporter)")
        self.logger.info("=" * 80)

        # Save to work_dir/tensorrt/ directory
        tensorrt_dir = os.path.join(self.config.export_config.work_dir, "tensorrt")
        os.makedirs(tensorrt_dir, exist_ok=True)

        # Determine output path based on ONNX file name
        if os.path.isdir(onnx_path):
            # For multi-file ONNX (shouldn't happen in standard export, but handle it)
            # Use the directory name or a default name
            output_path = os.path.join(tensorrt_dir, "model.engine")
        else:
            # Single file: extract filename and change extension
            onnx_filename = os.path.basename(onnx_path)
            engine_filename = onnx_filename.replace(".onnx", ".engine")
            output_path = os.path.join(tensorrt_dir, engine_filename)

        # Get sample input for shape configuration
        sample_idx = self.config.runtime_config.get("sample_idx", 0)
        sample = self.data_loader.load_sample(sample_idx)
        sample_input = self.data_loader.preprocess(sample)

        # Ensure tensor is float32
        if isinstance(sample_input, (list, tuple)):
            sample_input = sample_input[0]  # Use first input for shape
        if sample_input.dtype != torch.float32:
            sample_input = sample_input.float()

        # Merge backend_config.model_inputs into trt_settings for TensorRTExporter
        if hasattr(self.config, "backend_config") and hasattr(self.config.backend_config, "model_inputs"):
            trt_settings = trt_settings.copy()
            trt_settings["model_inputs"] = self.config.backend_config.model_inputs

        # Use provided exporter or create default
        if self._tensorrt_exporter is not None:
            exporter = self._tensorrt_exporter
        else:
            exporter = TensorRTExporter(trt_settings, self.logger)

        success = exporter.export(
            model=None,  # Not used for TensorRT
            sample_input=sample_input,
            output_path=output_path,
            onnx_path=onnx_path,
        )

        if success:
            self.logger.info(f"✅ TensorRT export successful: {output_path}")
            return output_path
        else:
            self.logger.error(f"❌ TensorRT export failed")
            return None

    # TODO(vivdf): check this, the current design is not clean.
    def get_models_to_evaluate(self) -> List[Tuple[str, str, str]]:
        """
        Get list of models to evaluate from config.
        Returns:
            List of tuples (backend_name, model_path, device)
        """
        backends = self.config.get_evaluation_backends()
        models_to_evaluate: List[Tuple[str, str, str]] = []

        for backend_name, backend_cfg in backends.items():
            if not backend_cfg.get("enabled", False):
                continue

            device = backend_cfg.get("device", "cpu")
            model_path = None
            is_valid = False

            if backend_name == "pytorch":
                model_path = backend_cfg.get("checkpoint")
                if model_path:
                    is_valid = os.path.exists(model_path) and os.path.isfile(model_path)
            elif backend_name == "onnx":
                model_path = backend_cfg.get("model_dir")
                # If model_dir is None, try to infer from export config
                if model_path is None:
                    work_dir = self.config.export_config.work_dir
                    onnx_dir = os.path.join(work_dir, "onnx")
                    save_file = self.config.onnx_config.get("save_file", "model.onnx")
                    multi_file = self.config.onnx_config.get("multi_file", False)  # Default to single file

                    if os.path.exists(onnx_dir) and os.path.isdir(onnx_dir):
                        # Check for ONNX files in work_dir/onnx/ directory
                        onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
                        if onnx_files:
                            if multi_file:
                                # Multi-file ONNX: return directory path
                                model_path = onnx_dir
                                is_valid = True
                            else:
                                # Single file ONNX: use the save_file if it exists, otherwise use the first ONNX file found
                                expected_file = os.path.join(onnx_dir, save_file)
                                if os.path.exists(expected_file):
                                    model_path = expected_file
                                else:
                                    model_path = os.path.join(onnx_dir, onnx_files[0])
                                is_valid = True
                        else:
                            if multi_file:
                                # Multi-file ONNX but no files found: still return directory
                                model_path = onnx_dir
                                is_valid = True
                            else:
                                # Try single file path
                                model_path = os.path.join(onnx_dir, save_file)
                                is_valid = os.path.exists(model_path) and model_path.endswith(".onnx")
                    else:
                        if multi_file:
                            # Multi-file ONNX: return directory even if it doesn't exist yet
                            model_path = onnx_dir
                            is_valid = True
                        else:
                            # Fallback: try in work_dir directly (for backward compatibility)
                            model_path = os.path.join(work_dir, save_file)
                            is_valid = os.path.exists(model_path) and model_path.endswith(".onnx")
                else:
                    # model_dir is explicitly set in config
                    multi_file = self.config.onnx_config.get("multi_file", False)
                    if os.path.exists(model_path):
                        if os.path.isfile(model_path):
                            # Single file ONNX
                            is_valid = model_path.endswith(".onnx") and not multi_file
                        elif os.path.isdir(model_path):
                            # Directory: valid if multi_file is True, or if it contains ONNX files
                            if multi_file:
                                is_valid = True
                            else:
                                # Single file mode: find the ONNX file in directory
                                onnx_files = [f for f in os.listdir(model_path) if f.endswith(".onnx")]
                                if onnx_files:
                                    # Use the save_file if it exists, otherwise use the first ONNX file found
                                    save_file = self.config.onnx_config.get("save_file", "model.onnx")
                                    expected_file = os.path.join(model_path, save_file)
                                    if os.path.exists(expected_file):
                                        model_path = expected_file
                                    else:
                                        model_path = os.path.join(model_path, onnx_files[0])
                                    is_valid = True
                                else:
                                    is_valid = False
                        else:
                            is_valid = False
                    else:
                        is_valid = False
            elif backend_name == "tensorrt":
                model_path = backend_cfg.get("engine_dir")
                # If engine_dir is None, try to infer from export config
                if model_path is None:
                    work_dir = self.config.export_config.work_dir
                    engine_dir = os.path.join(work_dir, "tensorrt")
                    multi_file = self.config.onnx_config.get("multi_file", False)  # Use same config as ONNX

                    if os.path.exists(engine_dir) and os.path.isdir(engine_dir):
                        engine_files = [f for f in os.listdir(engine_dir) if f.endswith(".engine")]
                        if engine_files:
                            if multi_file:
                                # Multi-file TensorRT: return directory path
                                model_path = engine_dir
                                is_valid = True
                            else:
                                # Single file TensorRT: use the engine file matching ONNX filename
                                onnx_save_file = self.config.onnx_config.get("save_file", "model.onnx")
                                expected_engine = onnx_save_file.replace(".onnx", ".engine")
                                expected_path = os.path.join(engine_dir, expected_engine)
                                if os.path.exists(expected_path):
                                    model_path = expected_path
                                else:
                                    # Fallback: use the first engine file found
                                    model_path = os.path.join(engine_dir, engine_files[0])
                                is_valid = True
                        else:
                            if multi_file:
                                # Multi-file TensorRT but no files found: still return directory
                                model_path = engine_dir
                                is_valid = True
                            else:
                                is_valid = False
                    else:
                        if multi_file:
                            # Multi-file TensorRT: return directory even if it doesn't exist yet
                            model_path = engine_dir
                            is_valid = True
                        else:
                            # Fallback: try in work_dir directly (for backward compatibility)
                            if os.path.exists(work_dir) and os.path.isdir(work_dir):
                                engine_files = [f for f in os.listdir(work_dir) if f.endswith(".engine")]
                                if engine_files:
                                    onnx_save_file = self.config.onnx_config.get("save_file", "model.onnx")
                                    expected_engine = onnx_save_file.replace(".onnx", ".engine")
                                    expected_path = os.path.join(work_dir, expected_engine)
                                    if os.path.exists(expected_path):
                                        model_path = expected_path
                                    else:
                                        model_path = os.path.join(work_dir, engine_files[0])
                                    is_valid = True
                                else:
                                    is_valid = False
                            else:
                                is_valid = False
                else:
                    # engine_dir is explicitly set in config
                    multi_file = self.config.onnx_config.get("multi_file", False)
                    if os.path.exists(model_path):
                        if os.path.isfile(model_path):
                            # Single file TensorRT
                            is_valid = (
                                model_path.endswith(".engine") or model_path.endswith(".trt")
                            ) and not multi_file
                        elif os.path.isdir(model_path):
                            # Directory: valid if multi_file is True, or if it contains engine files
                            if multi_file:
                                is_valid = True
                            else:
                                # Single file mode: find the engine file in directory
                                engine_files = [f for f in os.listdir(model_path) if f.endswith(".engine")]
                                if engine_files:
                                    # Try to match ONNX filename, otherwise use the first engine file found
                                    onnx_save_file = self.config.onnx_config.get("save_file", "model.onnx")
                                    expected_engine = onnx_save_file.replace(".onnx", ".engine")
                                    expected_path = os.path.join(model_path, expected_engine)
                                    if os.path.exists(expected_path):
                                        model_path = expected_path
                                    else:
                                        model_path = os.path.join(model_path, engine_files[0])
                                    is_valid = True
                                else:
                                    is_valid = False

            if is_valid and model_path:
                models_to_evaluate.append((backend_name, model_path, device))
                self.logger.info(f"  - {backend_name}: {model_path} (device: {device})")
            elif model_path:
                self.logger.warning(f"  - {backend_name}: {model_path} (not found or invalid, skipping)")

        return models_to_evaluate

    def run_verification(
        self, pytorch_checkpoint: Optional[str], onnx_path: Optional[str], tensorrt_path: Optional[str], **kwargs
    ) -> Dict[str, Any]:
        """
        Run verification on exported models using policy-based verification.
        Args:
            pytorch_checkpoint: Path to PyTorch checkpoint (reference)
            onnx_path: Path to ONNX model file/directory
            tensorrt_path: Path to TensorRT engine file/directory
            **kwargs: Additional project-specific arguments
        Returns:
            Verification results dictionary
        """
        verification_cfg = self.config.verification_config

        # Check master switches
        if not verification_cfg.get("enabled", True):
            self.logger.info("Verification disabled (verification.enabled=False), skipping...")
            return {}

        export_mode = self.config.export_config.mode
        scenarios = self.config.get_verification_scenarios(export_mode)

        if not scenarios:
            self.logger.info(f"No verification scenarios for export mode '{export_mode}', skipping...")
            return {}

        if not pytorch_checkpoint:
            self.logger.warning("PyTorch checkpoint path not available, skipping verification")
            return {}

        verification_cfg = self.config.verification_config
        num_verify_samples = verification_cfg.get("num_verify_samples", 3)
        tolerance = verification_cfg.get("tolerance", 0.1)
        devices_map = verification_cfg.get("devices", {}) or {}

        self.logger.info("=" * 80)
        self.logger.info(f"Running Verification (mode: {export_mode})")
        self.logger.info("=" * 80)

        all_results = {}
        total_passed = 0
        total_failed = 0

        for i, policy in enumerate(scenarios):
            ref_backend = policy["ref_backend"]
            # Resolve device using alias system:
            # - Scenarios use aliases (e.g., "cpu", "cuda") for flexibility
            # - Actual device strings are defined in verification["devices"]
            # - This allows easy device switching: change devices["cpu"] to affect all CPU verifications
            ref_device_key = policy["ref_device"]
            if ref_device_key in devices_map:
                ref_device = devices_map[ref_device_key]
            else:
                # Fallback: use the key directly if not found in devices_map (backward compatibility)
                ref_device = ref_device_key
                self.logger.warning(f"Device alias '{ref_device_key}' not found in devices map, using as-is")

            test_backend = policy["test_backend"]
            test_device_key = policy["test_device"]
            if test_device_key in devices_map:
                test_device = devices_map[test_device_key]
            else:
                # Fallback: use the key directly if not found in devices_map (backward compatibility)
                test_device = test_device_key
                self.logger.warning(f"Device alias '{test_device_key}' not found in devices map, using as-is")

            self.logger.info(
                f"\Scenarios {i+1}/{len(scenarios)}: {ref_backend}({ref_device}) vs {test_backend}({test_device})"
            )

            # Resolve model paths based on backend
            ref_path = None
            test_path = None

            if ref_backend == "pytorch":
                ref_path = pytorch_checkpoint
            elif ref_backend == "onnx":
                ref_path = onnx_path

            if test_backend == "onnx":
                test_path = onnx_path
            elif test_backend == "tensorrt":
                test_path = tensorrt_path

            if not ref_path or not test_path:
                self.logger.warning(f"  Skipping: missing paths (ref={ref_path}, test={test_path})")
                continue

            # Use policy-based verification interface
            verification_results = self.evaluator.verify(
                ref_backend=ref_backend,
                ref_device=ref_device,
                ref_path=ref_path,
                test_backend=test_backend,
                test_device=test_device,
                test_path=test_path,
                data_loader=self.data_loader,
                num_samples=num_verify_samples,
                tolerance=tolerance,
                verbose=False,
            )

            # Extract results for this specific comparison
            policy_key = f"{ref_backend}_{ref_device}_vs_{test_backend}_{test_device}"
            all_results[policy_key] = verification_results

            if "summary" in verification_results:
                summary = verification_results["summary"]
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                total_passed += passed
                total_failed += failed

                if failed == 0:
                    self.logger.info(f"  ✅ Policy {i+1} passed ({passed} comparisons)")
                else:
                    self.logger.warning(f"  ⚠️  Policy {i+1} failed ({failed}/{passed+failed} comparisons)")

        # Overall summary
        self.logger.info("\n" + "=" * 80)
        if total_failed == 0:
            self.logger.info(f"✅ All verifications passed! ({total_passed} total)")
        else:
            self.logger.warning(f"⚠️  {total_failed}/{total_passed + total_failed} verifications failed")
        self.logger.info("=" * 80)

        all_results["summary"] = {
            "passed": total_passed,
            "failed": total_failed,
            "total": total_passed + total_failed,
        }

        return all_results

    def run_evaluation(self, **kwargs) -> Dict[str, Any]:
        """
        Run evaluation on specified models.
        Args:
            **kwargs: Additional project-specific arguments
        Returns:
            Dictionary containing evaluation results for all backends
        """
        eval_config = self.config.evaluation_config

        if not eval_config.get("enabled", False):
            self.logger.info("Evaluation disabled, skipping...")
            return {}

        self.logger.info("=" * 80)
        self.logger.info("Running Evaluation")
        self.logger.info("=" * 80)

        models_to_evaluate = self.get_models_to_evaluate()

        if not models_to_evaluate:
            self.logger.warning("No models found for evaluation")
            return {}

        num_samples = eval_config.get("num_samples", 10)
        if num_samples == -1:
            num_samples = self.data_loader.get_num_samples()

        verbose_mode = eval_config.get("verbose", False)

        all_results: Dict[str, Any] = {}

        for backend, model_path, backend_device in models_to_evaluate:

            if backend in ("pytorch", "onnx"):
                if backend_device not in (None, "cpu") and not str(backend_device).startswith("cuda"):
                    self.logger.warning(
                        f"Unsupported device '{backend_device}' for backend '{backend}'. Falling back to CPU."
                    )
                    backend_device = "cpu"
            elif backend == "tensorrt":
                if backend_device is None:
                    backend_device = "cuda:0"
                if backend_device != "cuda:0":
                    self.logger.warning(
                        f"TensorRT evaluation only supports 'cuda:0'. Overriding device from '{backend_device}' to 'cuda:0'."
                    )
                    backend_device = "cuda:0"

            if backend_device is None:
                backend_device = "cpu"

            results = self.evaluator.evaluate(
                model_path=model_path,
                data_loader=self.data_loader,
                num_samples=num_samples,
                backend=backend,
                device=backend_device,
                verbose=verbose_mode,
            )

            all_results[backend] = results

            self.logger.info(f"\n{backend.upper()} Results:")
            self.evaluator.print_results(results)

        if len(all_results) > 1:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Cross-Backend Comparison")
            self.logger.info("=" * 80)

            for backend, results in all_results.items():
                self.logger.info(f"\n{backend.upper()}:")
                if results and "error" not in results:
                    if "accuracy" in results:
                        self.logger.info(f"  Accuracy: {results.get('accuracy', 0):.4f}")
                    if "mAP" in results:
                        self.logger.info(f"  mAP: {results.get('mAP', 0):.4f}")
                    if "latency_stats" in results:
                        stats = results["latency_stats"]
                        self.logger.info(f"  Latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
                    elif "latency" in results:
                        latency = results["latency"]
                        self.logger.info(f"  Latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
                else:
                    self.logger.info("  No results available")

        return all_results

    def run(self, checkpoint_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute the complete deployment workflow.

        Args:
            checkpoint_path: Path to PyTorch checkpoint (optional)
            **kwargs: Additional project-specific arguments

        Returns:
            Dictionary containing deployment results
        """
        results = {
            "pytorch_model": None,
            "onnx_path": None,
            "tensorrt_path": None,
            "verification_results": {},
            "evaluation_results": {},
        }

        export_mode = self.config.export_config.mode
        should_export_onnx = self.config.export_config.should_export_onnx()
        should_export_trt = self.config.export_config.should_export_tensorrt()

        # Resolve checkpoint / ONNX sources from config if not provided via CLI
        if checkpoint_path is None:
            checkpoint_path = self.config.export_config.checkpoint_path

        external_onnx_path = self.config.export_config.onnx_path

        # Check if we need model loading and export
        eval_config = self.config.evaluation_config
        verification_cfg = self.config.verification_config
        needs_onnx_eval = False
        if eval_config.get("enabled", False):
            models_to_eval = eval_config.get("models", {})
            if models_to_eval.get("onnx") or models_to_eval.get("tensorrt"):
                needs_onnx_eval = True

        requires_pytorch_model = False
        if should_export_onnx:
            requires_pytorch_model = True
        elif eval_config.get("enabled", False):
            models_to_eval = eval_config.get("models", {})
            if models_to_eval.get("pytorch"):
                requires_pytorch_model = True
        elif needs_onnx_eval and eval_config.get("models", {}).get("pytorch"):
            requires_pytorch_model = True
        elif verification_cfg.get("enabled", False) and should_export_onnx:
            requires_pytorch_model = True

        # Load model if needed for export or ONNX/TensorRT evaluation
        pytorch_model = None

        if requires_pytorch_model:
            if not checkpoint_path:
                self.logger.error(
                    "Checkpoint required but not provided. Please set export.checkpoint_path in config or pass via CLI."
                )
                return results

            self.logger.info("\nLoading PyTorch model...")
            try:
                pytorch_model = self.load_pytorch_model(checkpoint_path, **kwargs)
                results["pytorch_model"] = pytorch_model
            except Exception as e:
                self.logger.error(f"Failed to load PyTorch model: {e}")
                return results

        # Export ONNX if requested
        if should_export_onnx:
            if pytorch_model is None:
                if not checkpoint_path:
                    self.logger.error("ONNX export requires checkpoint_path but none was provided.")
                    return results
                self.logger.info("\nLoading PyTorch model for ONNX export...")
                try:
                    pytorch_model = self.load_pytorch_model(checkpoint_path, **kwargs)
                    results["pytorch_model"] = pytorch_model
                except Exception as e:
                    self.logger.error(f"Failed to load PyTorch model: {e}")
                    return results

            try:
                onnx_path = self.export_onnx(pytorch_model, **kwargs)
                results["onnx_path"] = onnx_path
            except Exception as e:
                self.logger.error(f"Failed to export ONNX: {e}")

        # Export TensorRT if requested
        if should_export_trt:
            onnx_source = results["onnx_path"] or external_onnx_path
            if not onnx_source:
                self.logger.error(
                    "TensorRT export requires an ONNX path. Please set export.onnx_path in config or enable ONNX export."
                )
                return results
            else:
                results["onnx_path"] = onnx_source  # Ensure verification/evaluation can use this path
                try:
                    tensorrt_path = self.export_tensorrt(onnx_source, **kwargs)
                    results["tensorrt_path"] = tensorrt_path
                except Exception as e:
                    self.logger.error(f"Failed to export TensorRT: {e}")

        # Get model paths from evaluation config if not exported
        if not results["onnx_path"] or not results["tensorrt_path"]:
            eval_models = self.config.evaluation_config.get("models", {})
            if not results["onnx_path"]:
                onnx_path = eval_models.get("onnx")
                if onnx_path and os.path.exists(onnx_path):
                    results["onnx_path"] = onnx_path
                elif onnx_path:
                    self.logger.warning(f"ONNX file from config does not exist: {onnx_path}")
            if not results["tensorrt_path"]:
                tensorrt_path = eval_models.get("tensorrt")
                if tensorrt_path and os.path.exists(tensorrt_path):
                    results["tensorrt_path"] = tensorrt_path
                elif tensorrt_path:
                    self.logger.warning(f"TensorRT engine from config does not exist: {tensorrt_path}")

        # Verification
        verification_results = self.run_verification(
            pytorch_checkpoint=checkpoint_path,
            onnx_path=results["onnx_path"],
            tensorrt_path=results["tensorrt_path"],
            **kwargs,
        )
        results["verification_results"] = verification_results

        # Evaluation
        evaluation_results = self.run_evaluation(**kwargs)
        results["evaluation_results"] = evaluation_results

        self.logger.info("\n" + "=" * 80)
        self.logger.info("Deployment Complete!")
        self.logger.info("=" * 80)

        return results
