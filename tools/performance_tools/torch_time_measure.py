import argparse
import functools
import os
import os.path as osp
import time

import numpy as np
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--batch-size", default=1, type=int, help="override the batch size in the config")
    parser.add_argument("--max-iter", default=200, type=int, help="maximum number of iterations to test")
    parser.add_argument("--warmup-iters", default=50, type=int, help="maximum number of iterations for warmup")

    args = parser.parse_args()
    return args


def wrapper(function_call, time_required, batch_size, max_iter, warmup_iters=10):
    @functools.wraps(function_call)
    def function(*args, **kwargs):
        start_time = time.perf_counter()
        result = function_call(*args, **kwargs)
        end_time = time.perf_counter()

        time_taken = end_time - start_time
        time_required.append(time_taken*1000)

        if len(time_required) >= max_iter + warmup_iters:
            mean_time = np.mean(time_required[warmup_iters:])
            std_dev = np.std(time_required[warmup_iters:])
            percentiles = np.percentile(time_required[warmup_iters:], [50, 80, 90, 95, 99])

            print("\nExecution Time Statistics:")
            print("-" * 40)
            print(f"Batch Size       : {batch_size}")
            print(f"Iterations       : {max_iter}")
            print(f"Mean Time        : {mean_time:.6f} ms")
            print(f"Standard Dev.    : {std_dev:.6f} ms")
            print(f"50th Percentile  : {percentiles[0]:.6f} ms")
            print(f"80th Percentile  : {percentiles[1]:.6f} ms")
            print(f"90th Percentile  : {percentiles[2]:.6f} ms")
            print(f"95th Percentile  : {percentiles[3]:.6f} ms")
            print(f"99th Percentile  : {percentiles[4]:.6f} ms")
            print("-" * 40)

            time_required.clear()
            exit(0)

        return result

    return function


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.batch_size = args.batch_size
    cfg.load_from = args.checkpoint

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    time_required = []
    test_function = runner.model.test_step
    runner.model.test_step = wrapper(
        test_function,
        time_required,
        args.batch_size,
        min(args.max_iter, len(runner.test_dataloader.dataset) - args.warmup_iters),
        args.warmup_iters,
    )
    runner.test()


if __name__ == "__main__":
    main()
