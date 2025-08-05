"""Script to compute analysis of T4 datasets."""
from pathlib import Path 
from typing import Any 

import argparse

import yaml 
from mmengine.config import Config
from mmengine.logging import print_log

from tools.dataset_preparation.enum import Task 
from tools.dataset_preparation.dataset.base.dataset_preparation_base import DatasetPreparationBase
from tools.dataset_preparation.dataset.detection3d.t4dataset_detection3d_preparation import \
  T4DatasetDetection3DPreparation


def parse_args():
    parser = argparse.ArgumentParser(description="Create data info for T4dataset")
    parser.add_argument(
      '--task',
      choices=['detection3d', 'detection2d', 'classification2d'],
      help='Choose a task for data preparation.',
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config for T4dataset",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="product version",
    )
    parser.add_argument(
        "--max_sweeps",
        type=int,
        required=False,
        help="specify sweeps of lidar per example",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="output directory of info file",
    )
    parser.add_argument(
        "--use_available_dataset_version",
        action="store_true",
        help="Will resort to using the available dataset version if the one specified in the config file does not exist.",
    )
    args = parser.parse_args()
    return args


def build_task(task: Task, config: Any, args: Any) -> DatasetPreparationBase:
    """ Build DataPreparation based on the task. """
    if task == Task.DETECTION3D:
        assert args.max_sweeps, f"max_sweeps must be set when the data preparation task is {Task.DETECTION3D}."
        dataset_preparation = T4DatasetDetection3DPreparation(
            root_path=Path(args.root_path),
            config=config, 
            info_save_path=Path(args.outout_dir),
            info_version=args.version,
            max_sweeps=args.max_sweeps,
            use_available_dataset_version=args.use_available_dataset_version,
        )
    else:
        raise ValueError(f"Task: {task} not supported yet!")
    
    return dataset_preparation 

def main():
    """Main enrtypoint to run the Runner."""
    # Load argparse
    args = parse_args()

    # load config
    config = Config.fromfile(args.config)

    # Build task
    dataset_preparation = build_task(
        task=Task[args.task],
        config=config,
        args=args
    )

    # Run dataset preparation
    dataset_preparation.run()

if __name__ == "__main__":
    main()