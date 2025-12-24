"""
GPU Resource Management Mixin for TensorRT Pipelines.

This module provides a standardized approach to GPU resource cleanup,
ensuring proper release of TensorRT engines, contexts, and CUDA memory.

Design Principles:
    1. Single Responsibility: Resource cleanup logic is centralized
    2. Context Manager Protocol: Supports `with` statement for automatic cleanup
    3. Explicit Cleanup: Provides `cleanup()` for manual resource release
    4. Thread Safety: Uses local variables instead of instance state where possible
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pycuda.driver as cuda
import torch

logger = logging.getLogger(__name__)


def clear_cuda_memory() -> None:
    """
    Clear CUDA memory cache and synchronize.

    This is a utility function that safely clears GPU memory
    regardless of whether CUDA is available.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class GPUResourceMixin(ABC):
    """
    Mixin class for pipelines that manage GPU resources.

    This mixin provides:
    - Standard `cleanup()` interface for resource release
    - Context manager protocol for automatic cleanup
    - Safe cleanup in `__del__` as fallback

    Subclasses must implement `_release_gpu_resources()` to specify
    which resources to release.

    Usage:
        class MyTensorRTPipeline(BaseDeploymentPipeline, GPUResourceMixin):
            def _release_gpu_resources(self) -> None:
                # Release TensorRT engines, contexts, CUDA buffers, etc.
                ...

    With context manager:
        with MyTensorRTPipeline(...) as pipeline:
            results = pipeline.infer(data)
        # Resources automatically cleaned up

    Explicit cleanup:
        pipeline = MyTensorRTPipeline(...)
        try:
            results = pipeline.infer(data)
        finally:
            pipeline.cleanup()
    """

    _cleanup_called: bool = False

    @abstractmethod
    def _release_gpu_resources(self) -> None:
        """
        Release GPU-specific resources.

        Subclasses must implement this to release their specific resources:
        - TensorRT engines and execution contexts
        - CUDA device memory allocations
        - CUDA streams
        - Any other GPU-bound resources

        This method should be idempotent (safe to call multiple times).
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """
        Explicitly cleanup GPU resources and release memory.

        This method should be called when the pipeline is no longer needed.
        It's safe to call multiple times.

        For automatic cleanup, use the pipeline as a context manager:
            with pipeline:
                results = pipeline.infer(data)
        """
        if self._cleanup_called:
            return

        try:
            self._release_gpu_resources()
            clear_cuda_memory()
            self._cleanup_called = True
            logger.debug(f"{self.__class__.__name__}: GPU resources released")
        except Exception as e:
            logger.warning(f"Error during GPU resource cleanup: {e}")

    def __enter__(self):
        """Context manager entry - return self for use in with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Destructor - cleanup as fallback if not done explicitly."""
        try:
            self.cleanup()
        except Exception:
            pass  # Suppress errors in destructor


class TensorRTResourceManager:
    """
    Context manager for TensorRT inference with automatic resource cleanup.

    This class manages temporary CUDA allocations during inference,
    ensuring they are properly freed even if an exception occurs.

    Usage:
        with TensorRTResourceManager() as manager:
            d_input = manager.allocate(input_nbytes)
            d_output = manager.allocate(output_nbytes)
            # ... run inference ...
        # All allocations automatically freed
    """

    def __init__(self):
        self._allocations: List[Any] = []
        self._stream: Optional[Any] = None

    def allocate(self, nbytes: int) -> Any:
        """
        Allocate CUDA device memory and track for cleanup.

        Args:
            nbytes: Number of bytes to allocate

        Returns:
            pycuda.driver.DeviceAllocation object
        """

        allocation = cuda.mem_alloc(nbytes)
        self._allocations.append(allocation)
        return allocation

    def get_stream(self) -> Any:
        """
        Get or create a CUDA stream.

        Returns:
            pycuda.driver.Stream object
        """
        if self._stream is None:
            self._stream = cuda.Stream()
        return self._stream

    def synchronize(self) -> None:
        """Synchronize the CUDA stream."""
        if self._stream is not None:
            self._stream.synchronize()

    def _release_all(self) -> None:
        """Release all tracked allocations."""
        for allocation in self._allocations:
            try:
                allocation.free()
            except Exception:
                pass
        self._allocations.clear()
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.synchronize()
        self._release_all()
        return False


def release_tensorrt_resources(
    engines: Optional[Dict[str, Any]] = None,
    contexts: Optional[Dict[str, Any]] = None,
    cuda_buffers: Optional[List[Any]] = None,
) -> None:
    """
    Release TensorRT resources safely.

    This is a utility function that handles the cleanup of various
    TensorRT resources in a safe, idempotent manner.

    Args:
        engines: Dictionary of TensorRT engine objects
        contexts: Dictionary of TensorRT execution context objects
        cuda_buffers: List of pycuda.driver.DeviceAllocation objects
    """
    # Release contexts first (they reference engines)
    if contexts:
        for name, context in list(contexts.items()):
            if context is not None:
                try:
                    del context
                except Exception:
                    pass
        contexts.clear()

    # Release engines
    if engines:
        for name, engine in list(engines.items()):
            if engine is not None:
                try:
                    del engine
                except Exception:
                    pass
        engines.clear()

    # Free CUDA buffers
    if cuda_buffers:
        for buffer in cuda_buffers:
            if buffer is not None:
                try:
                    buffer.free()
                except Exception:
                    pass
        cuda_buffers.clear()
