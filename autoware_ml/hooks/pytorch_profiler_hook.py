from enum import Enum

import numpy as np
from mmengine.dist.utils import get_rank
from mmengine.hooks import Hook
from mmengine.hooks.runtime_info_hook import DATA_BATCH
from mmengine.registry import HOOKS
from torch.cuda import current_device, max_memory_allocated
from torch.profiler import ProfilerActivity, profile


class ProfilerMode(Enum):
    """Profiler mode for PyTorch Profiler."""

    TRAIN = "train"
    TEST = "test"
    VAL = "val"


class PytorchProfilerHook(Hook):
    """A hook that profiles pytorch running stats for cuda."""

    priority = "NORMAL"

    def __init__(self, interval: int) -> None:
        super(PytorchProfilerHook, self).__init__()
        self._profiler = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        self._interval = interval
        self._profiler_running = False

        self._B = 8
        self._Kb = 1024
        self._Mb = 1024 * self._Kb
        self._Gb = 1024 * self._Mb
        self._time_us_to_ms = 1000

    def _start_profiler(self, runner) -> None:
        """Start the profiler if it is not running and the interval is met."""
        if self.every_n_train_iters(runner, self._interval) and not self._profiler_running:
            self._profiler.start()  # Start the profiler
            self._profiler_running = True

    def _step_profiler(self, runner, mode: ProfilerMode) -> None:
        """Collect data for the current step and stop profiling afterwards."""
        if not self._profiler_running:
            return

        self._profiler.step()  # Collect data for the current testing step

        # Finalize profiler data
        self._profiler.stop()
        self._profiler_running = False

        # Log profiling statistics
        cuda_time_total = 0.0
        self_cuda_time_total = 0.0
        max_cuda_memory = 0.0
        cuda_memory_usages = []

        key_averages = self._profiler.key_averages()
        for key_avg in key_averages:
            cuda_memory_usage = abs(key_avg.cuda_memory_usage)
            cuda_memory_usages.append(cuda_memory_usage)

            cuda_time_total += key_avg.cuda_time_total  # microseconds
            self_cuda_time_total += key_avg.self_cuda_time_total  # microseconds
            max_cuda_memory = max(cuda_memory_usage, max_cuda_memory)  # bits

        cuda_memory_usages_in_GB = np.asarray(cuda_memory_usages) / self._Gb / self._B
        total_cuda_memory = np.sum(cuda_memory_usages_in_GB)
        max_cuda_memory = np.max(cuda_memory_usages_in_GB)
        mean_cuda_memory = np.mean(cuda_memory_usages_in_GB)

        gpu_rank = get_rank()
        # It doesn't call reset_peak_memory_stats afterwards since it will be used in LoggerHook again
        max_cuda_memory_allocated = (
            max_memory_allocated(device=current_device()) / self._Gb
        )  # Convert from bytes to GB
        log_scalars = {
            f"{mode.value}/profiler/gpu:{gpu_rank}/total_cuda_duration (ms)": cuda_time_total / self._time_us_to_ms,
            f"{mode.value}/profiler/gpu:{gpu_rank}/total_cuda_self_duration (ms)": self_cuda_time_total
            / self._time_us_to_ms,
            f"{mode.value}/profiler/gpu:{gpu_rank}/total_cuda_stack_memory (GB)": total_cuda_memory,
            f"{mode.value}/profiler/gpu:{gpu_rank}/max_cuda_tensor_memory_allocated (GB)": max_cuda_memory_allocated,
            f"{mode.value}/profiler/gpu:{gpu_rank}/max_cuda_stack_memory (GB)": max_cuda_memory,
            f"{mode.value}/profiler/gpu:{gpu_rank}/mean_cuda_stack_memory (GB)": mean_cuda_memory,
        }
        runner.message_hub.update_scalars(log_scalars)


@HOOKS.register_module()
class PytorchTrainingProfilerHook(PytorchProfilerHook):
    """A hook that starts pytorch profiling for runtime and memory in training."""

    def __init__(self, interval: int) -> None:
        super(PytorchTrainingProfilerHook, self).__init__(interval=interval)

    def before_train_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None) -> None:
        """
        Starting profiler before training iteration if the mode is set to train and the interval is met.
        :param runner (Runner): The runner of the training process.
        :param batch_idx (int): The index of the current batch in the train loop.
        :param data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        self._start_profiler(runner)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        Collect profiler data if it's running.
        :param runner (Runner): The runner of the training process.
        :param batch_idx (int): The index of the current batch in the train loop.
        :param data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        self._step_profiler(runner, ProfilerMode.TRAIN)


@HOOKS.register_module()
class PytorchTestingProfilerHook(PytorchProfilerHook):
    """A hook that starts pytorch profiling for runtime and memory in testing."""

    def __init__(self, interval: int) -> None:
        super(PytorchTestingProfilerHook, self).__init__(interval=interval)

    def before_test_iter(self, runner, batch_idx, data_batch=None):
        """
        Starting profiler before training iteration if the mode is set to train and the interval is met.
        :param runner (Runner): The runner of the training process.
        :param batch_idx (int): The index of the current batch in the train loop.
        :param data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        self._start_profiler(runner)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        Collect profiler data if it's running.
        :param runner (Runner): The runner of the training process.
        :param batch_idx (int): The index of the current batch in the train loop.
        :param data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        self._step_profiler(runner, ProfilerMode.TEST)


@HOOKS.register_module()
class PytorchValidationProfilerHook(PytorchProfilerHook):
    """A hook that starts pytorch profiling for runtime and memory in validation."""

    def __init__(self, interval: int) -> None:
        super(PytorchTestingProfilerHook, self).__init__(interval=interval)

    def before_val_iter(self, runner, batch_idx, data_batch=None):
        """
        Starting profiler before training iteration if the mode is set to train and the interval is met.
        :param runner (Runner): The runner of the training process.
        :param batch_idx (int): The index of the current batch in the train loop.
        :param data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        self._start_profiler(runner)

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """
        Collect profiler data if it's running.
        :param runner (Runner): The runner of the training process.
        :param batch_idx (int): The index of the current batch in the train loop.
        :param data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
        """
        self._step_profiler(runner, ProfilerMode.VAL)
