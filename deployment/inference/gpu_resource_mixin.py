"""GPU Resource Management utilities for TensorRT Pipelines."""

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
    # Free the CUDA cache every N samples during long eval loops (GPU backends only).
    _gpu_cleanup_interval: int = 10

    @abstractmethod
    def _release_gpu_resources(self) -> None:
        """Release backend-specific GPU resources owned by the instance."""
        raise NotImplementedError

    def periodic_cleanup(self, sample_idx: int) -> None:
        """Free the CUDA cache every ``_gpu_cleanup_interval`` samples during long eval loops.

        Overrides the no-op :meth:`BaseInferencePipeline.periodic_cleanup` for every GPU-backed
        pipeline that mixes in this class (TensorRT), so CUDA cache growth over a long evaluation
        loop is bounded without each backend re-implementing the same guard.
        """
        if sample_idx > 0 and sample_idx % self._gpu_cleanup_interval == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self) -> None:
        """Release GPU resources once and clear CUDA caches (best effort)."""
        if self._cleanup_called:
            return

        try:
            self._release_gpu_resources()
            clear_cuda_memory()
            self._cleanup_called = True
            logger.debug("%s: GPU resources released", self.__class__.__name__)
        except Exception as e:
            logger.warning("Error during GPU resource cleanup: %s", e)

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

    def __init__(self) -> None:
        """Create an empty manager (no allocations and no stream)."""
        self._allocations: List[Any] = []
        self._stream: Optional[Any] = None

    def allocate(self, nbytes: int) -> Any:
        """Allocate `nbytes` on the device and track it for automatic cleanup."""
        allocation = cuda.mem_alloc(nbytes)
        self._allocations.append(allocation)
        return allocation

    @property
    def stream(self) -> Any:
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
) -> None:
    """Drop references to TensorRT engines/contexts so they are released.

    Contexts are cleared before engines (TensorRT requires execution contexts to be
    released before their parent engine). Destruction itself is refcount/GC-driven.
    """
    if contexts:
        contexts.clear()
    if engines:
        engines.clear()
