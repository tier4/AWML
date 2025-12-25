"""
GPU Resource Management utilities for TensorRT Pipelines.

Flattened from `deployment/pipelines/common/gpu_resource_mixin.py`.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pycuda.driver as cuda
import torch

logger = logging.getLogger(__name__)


def clear_cuda_memory() -> None:
    """Best-effort CUDA memory cleanup for long-running deployment workflows."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class GPUResourceMixin(ABC):
    """Mixin that provides idempotent GPU resource cleanup.

    Subclasses implement `_release_gpu_resources()` and this mixin ensures cleanup
    is called exactly once (including via context-manager or destructor paths).
    """

    _cleanup_called: bool = False

    @abstractmethod
    def _release_gpu_resources(self) -> None:
        """Release backend-specific GPU resources owned by the instance."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release GPU resources once and clear CUDA caches (best effort)."""
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class TensorRTResourceManager:
    """Helper that tracks CUDA allocations/stream for TensorRT inference.

    This is intentionally minimal: allocate device buffers, provide a stream,
    and free everything on context exit.
    """

    def __init__(self):
        """Create an empty manager (no allocations and no stream)."""
        self._allocations: List[Any] = []
        self._stream: Optional[Any] = None

    def allocate(self, nbytes: int) -> Any:
        """Allocate `nbytes` on the device and track it for automatic cleanup."""
        allocation = cuda.mem_alloc(nbytes)
        self._allocations.append(allocation)
        return allocation

    def get_stream(self) -> Any:
        """Return a lazily-created CUDA stream shared by the manager."""
        if self._stream is None:
            self._stream = cuda.Stream()
        return self._stream

    def synchronize(self) -> None:
        """Synchronize the tracked CUDA stream (if created)."""
        if self._stream is not None:
            self._stream.synchronize()

    def _release_all(self) -> None:
        """Free all tracked allocations and drop the stream reference."""
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
    """Best-effort release of TensorRT engines/contexts and CUDA buffers.

    This is defensive cleanup for cases where objects need explicit deletion and
    CUDA buffers need manual `free()`.
    """
    if contexts:
        for _, context in list(contexts.items()):
            if context is not None:
                try:
                    del context
                except Exception:
                    pass
        contexts.clear()

    if engines:
        for _, engine in list(engines.items()):
            if engine is not None:
                try:
                    del engine
                except Exception:
                    pass
        engines.clear()

    if cuda_buffers:
        for buffer in cuda_buffers:
            if buffer is not None:
                try:
                    buffer.free()
                except Exception:
                    pass
        cuda_buffers.clear()
