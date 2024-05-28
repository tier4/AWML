"""Download T4Dataset from DATA SEARCH with webauto command.

Setup a tool and configuration following <https://github.com/tier4/WebAutoCLI>.
"""

import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="list of t4dataset for train/val/test")
    parser.add_argument(
        "--project-id",
        type=str,
        choices=["prd_jt", "x2_dev"],
        required=True,
        help="project id of dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data",
        help="directory path for data to be downloaded",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    config_path = Path(args.config)
    assert config_path.exists() and config_path.is_file()

    out_dir = Path(args.out_dir)
    assert out_dir.exists() and out_dir.is_dir()

    with open(config_path) as f:
        data_splits = yaml.safe_load(f)

    required_keys = ["train", "val", "test"]
    assert isinstance(data_splits, dict) and all(
        [isinstance(data_splits[key], list) for key in required_keys]
    ), "config file must be a type of `dict[str, list]`"

    t4dataset_ids = sum([data_splits[key] for key in required_keys], [])

    command = "webauto data t4dataset pull --project-id {} --t4dataset-id {} --asset-dir {}"
    for t4dataset_id in t4dataset_ids:
        _command = command.format(args.project_id, t4dataset_id, args.out_dir)
        print(_command)
        subprocess.call(_command, shell=True)
