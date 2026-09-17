import csv
import json
import os
import time

from mmengine.dist import get_rank, get_world_size
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class IterTimingDumpHook(Hook):
    """Dump raw per-iteration ``data_time``/``time`` and per-epoch wall times.

    ``IterTimerHook`` only exposes windowed means through the logger. This hook
    writes ``timing_rank{r}.csv`` (one row per train iteration) and
    ``timing_rank{r}.json`` (per-epoch wall time and throughput) to
    ``work_dir`` on every rank, so dataloader stalls on any rank can be
    analysed after the run.
    """

    # Run after IterTimerHook (NORMAL) has updated the message hub.
    priority = "LOWEST"

    def __init__(self) -> None:
        self._rows = []
        self._epochs = []
        self._epoch_start = None

    def before_train_epoch(self, runner) -> None:
        self._epoch_start = time.perf_counter()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        hub = runner.message_hub
        self._rows.append(
            (
                runner.epoch + 1,
                runner.iter + 1,
                batch_idx,
                hub.get_scalar("train/data_time").current(),
                hub.get_scalar("train/time").current(),
            )
        )

    def after_train_epoch(self, runner) -> None:
        wall = time.perf_counter() - self._epoch_start
        loader = runner.train_dataloader
        iters = len(loader)
        samples = iters * loader.batch_size * get_world_size()
        self._epochs.append(
            dict(epoch=runner.epoch + 1, wall_time_s=wall, iters=iters, samples=samples, samples_per_s=samples / wall)
        )
        # Dump after every epoch so an interrupted run still leaves usable data.
        self._dump(runner)

    def _dump(self, runner) -> None:
        rank = get_rank()
        stem = os.path.join(runner.work_dir, f"timing_rank{rank}")
        with open(stem + ".csv", "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "iter", "batch_idx", "data_time", "time"])
            writer.writerows(self._rows)
        with open(stem + ".json", "w") as stream:
            json.dump(
                dict(
                    rank=rank,
                    world_size=get_world_size(),
                    batch_size=runner.train_dataloader.batch_size,
                    num_workers=runner.train_dataloader.num_workers,
                    epochs=self._epochs,
                ),
                stream,
                indent=2,
            )
