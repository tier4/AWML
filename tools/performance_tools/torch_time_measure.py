import argparse
import os
import os.path as osp
import time
import numpy as np
import functools
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--batch-size", default=1, type=int, help="override the batch size in the config")
    parser.add_argument("--max-iter", default=100, type=int, help="maximum number of iterations to test")
    args = parser.parse_args()
    return args


def wrapper(function_call, time_required, batch_size, max_iter):
    @functools.wraps(function_call)
    def function(*args, **kwargs):
        start_time = time.perf_counter()
        result = function_call(*args, **kwargs)
        end_time = time.perf_counter()
        
        time_required.append(end_time - start_time)
        if len(time_required) >= max_iter:
            print(f"Time required: {np.mean(time_required)} seconds @ batch size {batch_size}")
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
    runner.model.test_step = wrapper(test_function, time_required, args.batch_size, min(args.max_iter, len(runner.test_dataloader.dataset)))
    runner.test()


if __name__ == "__main__":
    main()
