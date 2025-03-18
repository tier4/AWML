import argparse
import logging
import time
from typing import List

import numpy as np
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from tqdm import tqdm

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMDet3D test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--batch-size", default=1, type=int, help="override the batch size in the config")
    parser.add_argument("--max-iter", default=200, type=int, help="maximum number of iterations to test")
    parser.add_argument("--warmup-iters", default=50, type=int, help="number of iterations for warmup")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.batch_size = args.batch_size
    cfg.load_from = args.checkpoint

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    time_required: List[float] = []
    test_dataloader = runner.test_dataloader
    test_step = runner.model.test_step

    logger.info("Starting model testing...")

    start_time = 0

    for i, data in tqdm(enumerate(test_dataloader), total=args.max_iter + args.warmup_iters):
        if i >= args.max_iter + args.warmup_iters:
            break

        _ = test_step(data)

        if start_time:
            end_time = time.perf_counter()
            time_taken = (end_time - start_time) * 1000  # Convert to milliseconds
            time_required.append(time_taken)
        start_time = time.perf_counter()

    # Compute and log statistics
    if len(time_required) > args.warmup_iters:
        time_required = time_required[args.warmup_iters :]
        mean_time = np.mean(time_required)
        std_dev = np.std(time_required)
        percentiles = np.percentile(time_required, [50, 80, 90, 95, 99])

        logger.info("\nExecution Time Statistics:")
        logger.info("-" * 40)
        logger.info(f"Batch Size       : {args.batch_size}")
        logger.info(f"Iterations       : {args.max_iter}")
        logger.info(f"Mean Time        : {mean_time:.6f} ms")
        logger.info(f"Standard Dev.    : {std_dev:.6f} ms")
        logger.info(f"50th Percentile  : {percentiles[0]:.6f} ms")
        logger.info(f"80th Percentile  : {percentiles[1]:.6f} ms")
        logger.info(f"90th Percentile  : {percentiles[2]:.6f} ms")
        logger.info(f"95th Percentile  : {percentiles[3]:.6f} ms")
        logger.info(f"99th Percentile  : {percentiles[4]:.6f} ms")
        logger.info("-" * 40)

    logger.info("Model testing completed successfully.")


if __name__ == "__main__":
    main()
